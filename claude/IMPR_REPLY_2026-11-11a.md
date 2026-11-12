# IMPR reply 2026-11-11 (a): answers and open items (IMPR, Kappa, tables)

**To:** the main thread, for Pedro. **From:** the IMPR session. **Written** 11-11 22:00; **updated 11-12 16:40.** New IMPR updates go at the top of this file, newest first.

## Update 11-12 16:40: the keys of your section 6 are in (`kappa.gap.*`, `demand.capsens.*`, `demand.findex.*`); the Panel B header is changed

**Files and the rebuild** (IMPR verified):
- **`make_paper_numbers.py`** `457a9439` → **`07769566`** (CR 0).
  - Four new collectors: `collect_netflow_gap`, `collect_cap_screen`, `collect_findex`, `collect_findex_series`.
  - One new helper, `_md5sums_stale`.
- **`make_tax_clock_table.py`** `125a157c` → **`d8b97f39`**.
- **The rebuild (16:32):** rc 0, **0 MISMATCH** (156 match). 419 → 427 entries; ok 348.
  - **No earlier value changed** except `reg.lambda.pfpj` (IMPR's all.min edit of 11-11).
  - The macros are identical in PAPER_NUMBERS and Drafts (`855088c0`).
- **Guards:**
  - every input must match its `MD5SUMS.txt`;
  - the panel must be fa38794d, and the gates G-A0, G-A1 and B-S3 must pass;
  - **the screen's production α must equal Table 5's current logit α to 1e-12.** If the logit is re-run, the capsens keys drop instead of going stale;
  - no environment override.
  - Each failure path was tested: it writes a note and builds no key.

**`kappa.gap.*`** (time deposits; A.1's 306,349 cells with a world unit):

| key | data | W0 | W3 |
|---|---|---|---|
| `kappa.gap.share.{uncapped, capped, truncated}` | 0.267 / 0.320 / 0.413 | | |
| `kappa.gap.median.uncapped.*` | 0.076 | 0.072 | 0.076 |
| `kappa.gap.median.capped.*` | 0.128 | **0.001** | 0.120 |
| `kappa.gap.median.truncated.*` | 0.101 | **0.003** | 0.010 |
| `kappa.gap.median.all.*` | 0.102 | 0.010 | 0.028 |

- `kappa.gap.lift.any` **1.020** (top decile 0.676, all cells 0.663).
- `kappa.gap.within.k4` **0.855** (carry-weighted).
- **One choice for you:** `kappa.gap.median.all.data` is 0.10178, over the 306,349 cells, while `kappa.k0.moving.median.k4` is 0.10211, over all 310,839.
  - They print the same at 3 dp and at `[p1]` (10.2), but differ at `[p2]` (10.18 against 10.21).
  - A one-line change makes `all.data` the key's value if you prefer.

**`demand.capsens.E3` / `.E4`.**
- **Leaves:** `cap098` (p1), `cap099` (production), `cap0995` (p2), `cap0999` (p4), `dropcap` (p3a), `share090` (s1), `share098` (s2), `cap09999` (p5), `cap099999` (p6).
  - Each has `alpha`, `p`, `elast`, `inside` (the median Σ_B s where φ̂ < 0.98) and `outside`.
  - p5 and p6 carry `inside` and `outside` only.
- **Usage:** for example, `\pn{demand.capsens.E3.cap0999.alpha}` → −0.092, and `.p` → 0.028.
- **The values** are the 14:55 table. The p5 and p6 inside shares are 0.0036 and 0.0005 (E3), and 0.0020 and 0.0002 (E4).

**`demand.findex.*`:**
- `demand.findex.constant` 0.77 (print it with `\pn[2]`);
- `demand.findex.series.2016` **0.694** and `.2024` **0.864**, one value per year, checked.
- Both definitions say that every row uses the constant today, because `process_specification`'s `keep_cols` drops the column, and that the fix is due at the next rebuild.

**Print formats.** Some values print 0.000 to 0.003 at the default 3 dp; use `[p2]` (percent) for them:
- `capsens.*.cap099999.inside`;
- `kappa.gap.median.capped.W0` and `.truncated.W0`.

**The Panel B header of `tab_tax_clock.tex`.** Only line 20 changed: "Panel B: smallest and largest value over the six cells of Panel A, by level of the CDI".
- `tax_clock_cost.csv` is byte-identical.
- CDI is defined in the papers (VF l.245).

**Still to come:**
- `demand.fs.efff.*`, from W1.1's numbers (16:20 section);
- `pix.aw1.*` and `pix.adv.own.close`, from W3 (16:15 section);
- `sleep.attn.*`, after W2 runs.
- They wait for your choices of what the text quotes. Pedro's go on new keys covers them.

## Update 11-12 16:35: the W2 plan (the stakes in the sleepiness step), read-only

**The plan:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\w2_plan_1112.md`. No repo file was edited, and nothing ran except light reads.

**First, the stakes F depends on the clustering.**
- P3's "F 782" (`pix.stakes`) is clustered by MCA; W1.1 reproduces it with 2,288 MCA clusters.
- **Clustered by state** (27 states, about 12 effective), the same regression gives **77**.
- With the lagged market structure as controls (as D22 says), it gives **18.5 in levels and 49.6 in logs**.
- **Identified across markets within a quarter (as D21 says), it falls to 7–8, which is weak.**
- So the paper's stakes F should lead with state clusters.

**What exists today.**
- E3 and E4 are run by `sleep_est_single.py`, with the states and instruments of `sleep_est_e2.py:165–199` and the kernels in `utils/sleep_links.py`.
- **The lagged Selic is already a column of the index** (`sleep_est_e2.py:167, 257`). Table 4 reports it: E3 −0.964 (not significant), E4 −1.029\*\*.
  - So **τ₀ is that number with the sign flipped**, and the split in your section 2 is a relabelling, with no code change.
- **The stakes, their first stage and the 99-draw placebo** exist only in diagnostic P3 (`diagnostics/diag_p_shifters.py:414–668`), which feeds `pix.stakes.*`. There is no stakes fit in the sleepiness step.

**The proposed change** (a new driver; no production edit, so today's results cannot move):
- **The driver:** `sleep_attention_stakes.py` reuses the production kernels unchanged and writes only to `Rout/attention_stakes/`.
  - A gate first refits the current E3/E4 and checks that it reproduces the stored fit.
- **The stakes enter the index as one more state,** instrumented by P3's S̃ through a control-function residual that also sits in the index.
  - Recommended: take market and quarter means out of the stakes. Their national path tracks the lagged Selic (correlation 0.85), so removing the quarter means keeps τ₀ comparable with Table 4.
  - Controls in both stages: the lagged market structure and the lagged mean balance.
- **Reporting:**
  - τ₀ and τ₁ as average marginal effects with the sign flipped, with the two-stage wild-cluster bias-corrected intervals Table 4 uses. The link-only bootstrap would give them identical t-statistics, so it cannot be used.
  - **The placebo:** each bank's actual and cost-fitted spreads are permuted together across banks within the quarter, with market structure and balance fixed. 99 draws, not one.
  - **Extras:** a reduced-form test of τ₁ = 0 that is robust to a weak instrument; the effective F under state and MCA clusters; and a permutation test.
- **Keys:** `sleep.attn.*` (τ₀, τ₁, the first stage, the placebo, the reduced form, the sample), through a new collector.

**Cost** (estimates; the smokes measure them):

| part | locally | Bouchet |
|---|---|---|
| point fits, first stage, gates | about 2.5–4 h, 6–10 GB | — |
| intervals, B = 999, four fits | 3–5 h each with 4 workers, 12–20 GB | about 10–15 min each on 64 cpus |
| placebo, 99 draws × 2 routines | 13–26 h | 20–40 min as an array |

**Decisions for Pedro** (the agent's recommendation in brackets; IMPR agrees):
1. **Where τ₁ is identified:** within a market over time, or across markets within a quarter (D21) [within; D21 then needs new wording].
2. **Level or log stakes:** [fix the rule before seeing τ₁: levels if the state-clustered F is at least 23.1, else logs; today the rule picks logs].
3. **Take quarter means out of the stakes** [yes].
4. **Drop markets with fewer than two banks taking time deposits** [yes; about 3 % of B deposits, with the baseline refitted on the same sample].
5. **The placebo:** 99 draws [yes].
6. **Where to run:** the point fits locally, and the intervals and the placebo on Bouchet [yes]. The alternative, everything locally, takes about two nights.
7. **Extras:** the reduced-form test [yes]; the MCA sensitivity and the first-stage-error checks [optional].
8. **The paper's stakes F:** lead with state clusters, with MCA and the permutation beside it [yes].
9. **The `sleep.attn.*` keys:** a new collector. It is covered by Pedro's go on new keys once the results exist.

**Biggest risks:**
- the weak instrument once controls and geographic clusters are in place;
- the index already identifies about 3 directions well, and gains 5 more columns;
- the bootstrap runs on about 6 effective clusters;
- the stakes share the spreads with the existing spread correction and the rival-spread instrument.
- **So expect weak-instrument language,** unless the log form holds up. Lead with the reduced-form test and the placebo, not the point estimate.

## Update 11-12 16:25: W4.1 (feasibility of the availability index): new money is nearly all of the awake pool, so only advertising is well identified

All of it is read-only, under the revised model of 15:52 (γ_P for rival money, γ_P∅ for new money).
- **The plan:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\w4_plan_1112.md`.
- **Scripts and outputs:** `scratchpad\w4_1112\` (`w4_feasibility.py`/`.json`, `w4_adv_margins.py`, `w4_bnew_by_year.py`).
- Each run took under 40 s and under 0.9 GB.

**Your first-pass form: the algebra is right, under three conditions.**
1. tastes at the mean utility;
2. one common denominator for every consideration set;
3. pools that use up the market size, which holds in 99.98 % of market-quarters.
- The second launch term changes nothing.
- **Its first-order error,** ε_j ≈ Σ_s (w_js − b_s)·U_s − e_j·Σ_{s≠j} w_js·(1 − π_{j|s}), is dominated by bank j's own size times its unavailability.
  - On the data: median −0.014, p10 −0.076, p1 −0.29, within-quarter SD 0.05.
  - That is about the size of the variation that identifies γ_P∅, and more than the variation for γ_P.
  - Its correlation with those regressors is small (−0.02 and +0.12), and **ε can be computed from the observed shares and added to the form.**
- **Two further approximations come from the data:**
  - the prep divides by the capped φ, while the active balance uses the uncapped φ;
  - D-held money is classed as new money.

**W4.1: new money is nearly all of the awake pool.**

| | base cap 0.99 | cap 0.999 |
|---|---|---|
| new money's share of the market size, median destination cell | **95 %** | 99 % |
| the same, deposit-weighted median | 78 % | 93 % |
| new money's share of W at availability 0.5 | 93 % | 98.7 % |
| own and rival money's share of W | 2 % and 4 % | 0.4 % and 0.8 % |
| cells where rival pools exceed 10 % of the market size | 15 % | 1 % |

**How much each parameter can move ln W, per unit, against the residual of the logit's left-hand side** (SD 2.2 within quarter, 0.82 within two-way):

| parameter | share of the residual SD | reading |
|---|---|---|
| γ1, advertising | **61–77 %** | well determined, **but through new money:** it rests on advertising being excluded from utility, not on the own-versus-rival contrast |
| γ_P, Pix for rival money | **about 1.2 %** | weak at best; the top decile of cells only; half that at cap 0.999 |
| γ_P∅, Pix for new money | 2.1–2.4 % | identified by the 2020Q4 level break, which every other shock of that quarter shares, unless quarter effects and the dose are added. With quarter effects, it rests on a post-launch spread of 0.078 (0.055 at cap 0.999) |
| κ^c, the tax clock | 0.2–0.35 % | **not identified** |

- **The logit carries no fixed effects** (Q7.1), so nothing absorbs the launch.
- **The two Pix terms overlap:** their regressors correlate −0.49 within quarter at availability 0.5, and −0.78 at 0.2.
- **Advertising:** 17 % of its variance is within bank. Of that, 96 % comes from 34 banks switching between zero and positive spending.
- **The tax clock's shifter TD** is 88 % between bank-market cells. BB's and Caixa's type 4 includes judicial deposits.
- **Branches:** 3 % of their variance is within cells, so γ2 = 0.
- **The market size trends.**
  - It uses one anchor for all quarters, so B deposits over the market size rise from 3.6 % to 10.2 % over the window.
  - The rival share then rises after 2020 for a mechanical reason, which a misspecified W would load on γ_P.
- **The root cause** is that the market size is about 30 times the B stocks. The full share system with consideration sets would have the same problem.

**Q7.3: the inputs exist; the shares and the rival sum are new.**
- **What exists:**
  - K2 computes the accrued stock by bank, market and quarter (`diag_k2_netflow_simulation.py` l.398–412) and the market size with the outside pool (l.433–447);
  - K0 saves new money over the market size in `Rout/k0_outside_pool.csv`.
- **What is new:** the shares b are one division each. The rival sum, which varies by source bank, is a pairwise sum over about 1.1 million pairs.

**The W4.2 plan** (a new script, `diag_w4_availability_sim.py`, on Pedro's go):
- **The world:** K2's context with exact consideration-set shares (by enumeration, and simulated in the few large markets), and D always available.
- **The truth:** γ1 0.17, γ_P 1, κ^c 1, γ_P∅ ∈ {0, 1}.
- **Fits:** the first-pass form and the corrected form, side by side.
- **Run A:** no noise, full sample, about 15–20 min; it gives the bias of the approximation.
- **Run B:** noise resampled from the logit residual, 300 markets, 50 replications, about 35 min; it gives bias and Monte Carlo SD.

**The W4.3 plan** (after W4.2):
- **The estimator:** the logit's 2SLS for the linear part, and Levenberg–Marquardt for the six index parameters.
- **Six variants:** no effects; quarter effects; the dose; lagged advertising as instrument; weighted by own stock; cap 0.999.
- **Inference:** the score wild cluster bootstrap by conglomerate.
- **Keys:** `avail.idx.{g1,gP,gPn,gP_minus_gPn,kc}.{est,se,lo,hi}.<variant>`. About 20–30 min locally.

**Decisions for Pedro** (IMPR's recommendation in brackets):
- **W-a:** W4.2 before W4.3 [yes].
- **W-b:** fit the corrected form beside the first-pass one [yes].
- **W-c:** D rows outside the index [yes].
- **W-d:** fix in advance what counts as "showing life" [yes].
- **W-e:** if W4.2 confirms the table, report only γ1 and γ_P∅ (with quarter effects and the dose), and **state the switching-cost difference γ_P − γ_P∅ and κ^c as not identified in this form** [recommended].
- **W-f:** let the market size grow with deposits, i.e. revisit the market size [Pedro's model decision; it is the other route to identifying the own and rival terms].


### W4 plan, full text (`scratchpad\w4_plan_1112.md`, md5 `f04a49cf`, written by the W4 agent 11-12 16:20; carried here verbatim, headings demoted)

### W4: the availability index. Algebra check, W4.1 feasibility table, plans for W4.2 and W4.3, and Q7.3

IMPR planning subagent, 2026-11-12, 15:29 to 16:2x (shell clock). It follows `claude/MAIN_REPLY_2026-11-12.md` as revised at 15:52: the launch enters both indices, γ_P for rival money and γ_P∅ for new money, and the switching-cost change is γ_P − γ_P∅.

**Nothing was changed.** No repo file was edited and nothing was deleted. I ran four light scripts, each under 40 s and under 0.9 GB. They are in `scratchpad\w4_1112\`:
- `w4_feasibility.py` → `w4_feasibility.json`
- `w4_adv_margins.py` → `.log`
- `w4_bnew_by_year.py` → `.log`
- one inline read of the prep (three columns, pyarrow)

**Tags.** [V-code] = read in the code; [V-data] = read in a data file or output, or computed by those scripts; [inf] = my inference.

**Inputs, all read-only:**
- the a216 K2 context `scratchpad\kappa_1109\k2n_ctx_fa38794d_a216.pkl` (md5 `8ba340ae`): panel fa38794d, production E3 frame `0dd20428`;
- the cap-0.999 E3 frame `C:\Users\pedro\AppData\Local\Temp\capsens_1111\p4\demand_prep\demand_3_index_spec_12.parquet`, of which only `M_mt` was used (`a1ab2f08`);
- `Rout/p2_advertising_conglomerate_quarter.csv` (`96786751`) and `Rout/p2_branches_conglomerate_mca_quarter.csv` (`9fae36ce`), both from P2 on the d01b8bae panel.

---

#### 0. Short answers

1. **The algebra is right, under three conditions:**
   - tastes at the mean utility (μ = 0);
   - a common denominator for every consideration set;
   - pools that exhaust the market size.
   Adding the second launch term changes nothing in it.
   - **Its error.** To first order it is ε_j ≈ Σ_s (w_js − b_s) U_s − e_j Σ_{s≠j} w_js (1 − π_{j|s}), defined in 1.3. On the data at availability 0.5 it has median −0.014, 10th percentile −0.076, 1st percentile −0.29, and SD 0.05 within quarter.
   - **What that means.** The error is larger than the variation that identifies γ_P and about equal to the variation that identifies γ_P∅. Its correlation with those regressors is small (−0.02 and +0.12). It can be computed from the data and added to the form.
2. **New money is nearly all of the awake pool. Plainly:**
   - **Base cap, typical cell.** At the median destination cell, new money is 95 % of the market size. At availability 0.5 it is 93 % of W; own money is 2 % and rival money 4 %.
   - **Cap 0.999.** New money is 99 % of the market size and 98.7 % of W.
   - **Where the money is,** new money is less dominant: 78 % at the deposit-weighted median under the base cap.
   - **What this does to each parameter:**
     - **κ^c is not identified in practice:** its regressor moves ln W by 0.2 to 0.35 % of the residual SD per unit.
     - **γ_P is at best weakly identified,** from the top decile of cells: 1.2 to 1.3 % of the residual SD per unit, and half that at cap 0.999.
     - **γ1 is well determined** (61 to 77 % of the residual SD), but through new money. It therefore rests on the exclusion of advertising from utility, not on the own-against-rival contrast.
     - **γ_P∅ is identified, but in the logit as it stands only by the level break at 2020Q4.** That break is shared with every other shock of that quarter.
3. **The logit carries no fixed effects at all:** no constant, no quarter, no bank and no market effects [V-code: `blp_logit.jl` 25–29, 93–94, 143–149]. So nothing absorbs the launch indicator.
   - **With quarter effects added:** γ_P∅ rests on a cross-sectional spread of 0.05 per unit (0.08 among post-launch cells), against a residual SD of 2.2.
   - **The two Pix regressors are collinear** within quarter: correlation −0.49 at availability 0.5, −0.78 at 0.2.
   - **The dose** (launch × Pix use) would give γ_P∅ cross-market variation that quarter effects do not absorb.
4. **Branches:** 3 % of the variance of ln(1 + branches) is within cells. Keep γ2 = 0.
5. **Q7.3.** The pieces exist: K2's accrued stock by source bank-market-quarter, the market size M̂, and K0's outside pool. The shares b against M̂ are computed nowhere but are one division each. The source-specific rival sum Σ_{j′≠j} b_j′ Ã_{j|j′} is new: a pairwise sum within the market, 1.1 million pairs.
6. **IMPR's recommendation.**
   - Run W4.2 before W4.3, about 1 h locally. It measures, with noise of the data's size, how precisely the first-pass form recovers each parameter.
   - If it confirms section 2, Pedro faces a model decision before W4.3 (section 6).
   - The identification problem comes from the size of the market size against the B stocks, not from the first-pass approximation. So the full share system would not remove it.

---

#### 1. The algebra of the first-pass form

##### 1.1 Derivation, with both launch terms

Fix a market-quarter (subscripts mt dropped). Sources are s ∈ {j′ : B banks in m} ∪ {∅}.
- Availability of institution l ≠ s:
  - π_{l|j′} = Ã_{l|j′} = Λ(γ0 + γ1 adv_l + γ2 br_l + γ_P Post − κ^c TD_j′);
  - π_{l|∅} = A^∅_l = Λ(γ∅ + γ1 adv_l + γ2 br_l + γ_P∅ Post).
- The source and the outside option are always available. Availability is independent across l (eq. 13, the form of Goeree).
- Write E_l = Σ_k exp(δ_lk) and D = 1 + Σ_{all l} E_l, the full-set denominator.

**The model's active share.** Eq. (16) gives Dep^Act_jk = (1 − φ)[Σ_j′ Stock_j′ s_{jk|j′} + Out s_{jk|∅}], with Out = M − Σ_j′ Stock_j′. Dividing by (1 − φ)M:

  s^Act_jk = Σ_s b_s s_{jk|s},  s^Act_0 = Σ_s b_s s_{0|s},  b_j′ = Stock_j′/M,  b_∅ = Out/M,  Σ_s b_s = 1.

**The approximation.** Take μ = 0 and replace every set's denominator D_C by D. Then
- s_{jk|s} ≈ π_{j|s} exp(δ_jk)/D, with π_{j|j} = 1;
- s_{0|s} ≈ 1/D for every source.

Hence

  ln s^Act_jk − ln s^Act_0 ≈ δ_jk + ln W_j,  W_j = b_j + Σ_{j′≠j} b_j′ Ã_{j|j′} + b_∅ A^∅_j.  ✓

The second launch term only changes A^∅. The algebra is unchanged. [V-code: eq. 13 and 16 in `paper_edits_2026-11-12.tex` 92–132; the prep's shares are `Dep_Act / ((1 − φ) M)`, `sleep_demand_prep_link.py` 467–468]

##### 1.2 Conditions under which it holds
- **C1. μ = 0, tastes at the mean utility.** Under random coefficients each set's share is an integral, and W does not factor out.
- **C2. Common denominator.** Unavailable institutions must remove little from the denominator. This needs a dominant outside option, or availability near one (1.3).
- **C3. Pools exhaust M:** Out ≥ 0, so Σ b = 1.
  - It holds in 99.98 % of market-quarters under the base cap, and in all of them at 0.999 [V-data].
  - Where it fails, b_∅ is clipped, and −ln Σ_s b_s belongs on the right-hand side.
- **C4. The model's own assumptions:** independence across institutions, and each source sharing the market's tastes, so no sorting (eq. 13's text).
- **Two things the data add on top.**
  - **The capped φ.** The prep divides by (1 − φ capped) while `Dep_Act` uses the uncapped φ (`sleep_demand_prep_link.py` 359–360 and 534–535) [V-code]. That puts a market-quarter factor on the B shares wherever the cap binds: 54 % of B rows at 0.99, 35 % at 0.999 [V-data: `frame_diag.json`].
  - **D-held money is classed as new money.** D institutions have no stock by market, so their customers' money sits in Out (K2's docstring, lines 13–14) [V-code]. It then faces A^∅ instead of the rivals' index with the Pix and clock terms.
    - The amount is small: D deposits are 0.04 % of the B market size in 2016Q1 and 1.7 % in 2024Q4 [V-data, prep].
    - But it is the part of new money that should carry γ_P [inf].

##### 1.3 How far the form is from the exact share with consideration sets

Exact with μ = 0: s_{jk|s} = exp(δ_jk) π_{j|s} E[1/D_C | j ∈ C; s] and s_{0|s} = E[1/D_C; s].

Write e_l = E_l/D for institution l's share of the full-set denominator, and U_s = Σ_{l≠s} (1 − π_{l|s}) e_l for the expected share of the denominator that source s misses. Expanding 1/D_C to first order in e:

  ln s^Act_jk − ln s^Act_0 = δ_jk + ln W_j + ε_j,  ε_j ≈ Σ_s (w_js − b_s) U_s − e_j Σ_{s≠j} w_js (1 − π_{j|s}),  w_js = b_s π_{j|s}/W_j.

- **The first term** is a composition effect. The money entering j has a different mix of sources from the market's pools, and sources differ in how much of the menu they miss.
- **The second term** is j's own size times its unavailability.
- **ε is negative in nearly every cell** (its 99th percentile is −0.0001). Fewer available rivals raise the outside option's share more than j's share.

**On the data**, at the central parameters (availability 0.5, γ1 0.17, γ_P 1, γ_P∅ 0, κ^c 1), with the D products as one block at the same availability [V-data, `w4_feasibility.json`, `approx_error_first_order`]:

| | median | p10 | p1 | mean | SD | SD within quarter | SD within two-way |
|---|---|---|---|---|---|---|---|
| ε, base cap | −0.014 | −0.076 | −0.287 | −0.031 | 0.053 | 0.051 | 0.028 |
| ε, base cap, availability 0.2 | −0.024 | −0.130 | −0.473 | −0.053 | 0.087 | 0.083 | |
| ε, cap 0.999 | −0.013 | −0.077 | −0.301 | −0.032 | 0.055 | 0.053 | 0.030 |

- **The second term dominates:** median −0.012 against −0.001 for the first.
- **The inside mass** of the full-availability logit, 1 − s_0, by market-quarter: median 0.136, 90th percentile 0.52, 95th 0.79. In 10.5 % of market-quarters it is above 0.5, and there the first-order expansion is itself unreliable.
- **Correlation of ε with the identifying regressors** within quarter (base, central): γ_P −0.025, γ_P∅ +0.118, γ1 −0.043, κ^c −0.013.
  - One-regressor bias implied [inf]: about −0.05 on γ_P, +0.11 on γ_P∅ and −0.002 on γ1, per unit.
  - At availability 0.2 the correlation with the γ_P∅ regressor rises to 0.20.

**Reading.** As noise the error is small: 0.05 against a residual SD of 2.2. But it is of the size of the variation that identifies γ_P (0.03 per unit) and γ_P∅ (0.05). So it matters for exactly the parameters the section cares about [inf].

**A corrected form is cheap.** e_l ≈ s^Act_l/W_l(θ) is observed given θ, so ε_j(θ) can be computed and added to the estimating equation. That makes it exact to second order in the inside mass. Recommend it as the second form in W4.2 and W4.3 [inf].

---

#### 2. W4.1 feasibility table

The units are destination bank-market-quarters: B units with a product on the menu at t, 295,977 of them. The pools are the accrued stocks at t − 1 over all types, as K2 builds them (`diag_k2_netflow_simulation.py` 398–412), divided by the market size M_mt. b_∅ = 1 − Σ stocks/M [V-code; V-data].

##### 2.1 The pools' shares of the market size

| share | base cap 0.99, unweighted p10 / p50 / p90 | base, weighted by own stock, p50 | cap 0.999, unweighted p10 / p50 / p90 | cap 0.999, stock-weighted p50 |
|---|---|---|---|---|
| own, b_jmt | 0.0012 / 0.011 / 0.041 | 0.037 | 0.0002 / 0.0022 / 0.0083 | 0.012 |
| rivals' total | 0.000 / 0.031 / 0.127 | 0.166 | 0.000 / 0.006 / 0.026 | 0.058 |
| new money, b_∅ | 0.840 / 0.954 / 0.994 | 0.784 | 0.967 / 0.991 / 0.999 | 0.927 |

[V-data]

- **By market-quarter,** the median b_∅ is 0.970 (base) and 0.994 (0.999). Weighted by M, the aggregate is 0.909 and 0.980.
- **Destination cells with rivals' total above:**

  | threshold | base | 0.999 |
  |---|---|---|
  | 0.05 | 35 % | 3.1 % |
  | 0.10 | 15 % | 1.1 % |
  | 0.25 | 2.5 % | 0.2 % |

- **b_∅ below 0.9:** 22 % of cells (base), 1.3 % (0.999).
- **The cap at 0.999 raises the anchor** r̂ from 82,912 to 415,940 [V-data: `frame_diag.json`]. That multiplies M about five-fold outside the floor-bound markets and drives b_∅ toward one.
- **A trend through the window** [V-data, `w4_bnew_by_year.log`; V-code, prep 405–424: one anchor r̂ for all quarters]:
  - the median b_∅ falls from 0.967 (2016) to 0.954 (2020) and 0.938 (2024);
  - the rivals' 90th percentile rises from 0.094 to 0.105 and then 0.165, with the step between 2020 and 2021;
  - the cause is that M̂ uses one anchor for all quarters while deposits grow: B deposits over M̂ go from 3.6 % in 2016Q1 to 7.7 % in 2020Q4 and 10.2 % in 2024Q4.
  - So the rival share of W rises after the launch for a mechanical reason, and γ_P would pick that up if W is misspecified [inf].

##### 2.2 The composition of W

At availability 0.5 for rivals and for new money (base; cap 0.999 in brackets) [V-data]:
- **own:** median 0.021, 90th percentile 0.075 (0.004 and 0.016);
- **rivals:** median 0.039, 90th percentile 0.157 (0.008 and 0.034);
- **new money:** median 0.932, 10th percentile 0.781 (0.987 and 0.952).

At availability 0.2, new money's median share is 0.889 (0.977); at 0.8 it is 0.948 (0.990).

##### 2.3 The shifters

- **Advertising, ln(1 + real advertising), B conglomerate-quarters** (69 conglomerates, 2,220 observed quarters) [V-data]:
  - within-bank share of the variance 17.0 % (SD 2.86); 16.6 % net of quarter effects as well;
  - **96 % of the within-bank sum of squares comes from 34 banks that switch between zero and positive spending;**
  - the 27 banks that always advertise contribute 4 %: within-bank SD 0.93 log points, median four-quarter change 0.37, 90th percentile 1.25;
  - 8 banks never advertise;
  - on destination units, 15.7 % of a variance of 3.37 is within bank×market and market×quarter effects;
  - coverage on destination units is 99.997 %.
  - **So with bank effects, γ1 rests mostly on on/off switches of small banks.** Without them, as in the logit, it rests on the cross-section of advertising (83 % of the variance) [inf].
- **TD of the source** (type-4 share of the accrued stock; 324,401 source units) [V-data]:
  - median 0.16, stock-weighted median 0.59;
  - variance 0.0555, of which 87.7 % is between bank×market cells and 12.3 % within (SD 0.083); 7.5 % is within cells net of market-quarter means;
  - BB's and Caixa's type 4 includes judicial deposits, which inflate their TD [V-data: K0's `judicial` flag; memory].
- **Branches** [V-data; P2 note]:
  - median count 1 (90th percentile 2, 99th 19); 14 % of cells change their count;
  - 3.0 % of the variance of ln(1 + branches) is within cells;
  - **keep γ2 = 0.**

##### 2.4 How much ln W each parameter moves

The table gives the Jacobian columns of ln W at the central parameters, as SDs per unit of each parameter, over destination units [V-data]. The noise scale is δ, the logit's left-hand side, over 625,677 B product rows: SD 2.65 overall, 2.20 within type × quarter, 0.82 within bank×market×type and market×quarter×type.

| parameter | SD overall | SD within quarter | SD within two-way | within quarter ÷ 2.20 | within two-way ÷ 0.82 |
|---|---|---|---|---|---|
| γ1 (advertising) | 1.357 | 1.342 | 0.636 | 61 % | 77 % |
| γ_P (Pix, rival money) | 0.030 | 0.028 | 0.0095 | 1.3 % | 1.2 % |
| γ_P∅ (Pix, new money) | 0.220 | 0.052 | 0.017 | 2.4 % | 2.1 % |
| κ^c (the clock) | 0.008 | 0.0078 | 0.0019 | 0.35 % | 0.23 % |
| γ2 = 0.5 (branches; the Δ, not per unit) | 0.078 | 0.078 | 0.016 | | |
| **cap 0.999:** γ_P | 0.015 | 0.0145 | 0.005 | 0.7 % | 0.6 % |
| **cap 0.999:** γ_P∅ | 0.241 | 0.036 | 0.020 | 1.6 % | 2.5 % |
| **cap 0.999:** κ^c | 0.0035 | 0.0034 | 0.001 | 0.15 % | 0.12 % |

**Among post-launch cells:**
- **The γ_P column:** mean 0.023, SD 0.042; median 0.012, 90th percentile 0.050, 99th 0.19.
- **The γ_P∅ column:** mean 0.43, SD 0.079; 10th percentile 0.34, 90th 0.52.
- **Quarter effects absorb** 0.4 % of the first column's variance among post cells, but 94 % of the second's over all cells, since that is the jump at the launch.
- **Collinearity of the two Pix columns** within quarter (post cells): −0.49 at availability 0.5, −0.78 at 0.2, +0.03 at 0.8. At cap 0.999: −0.04, −0.63 and +0.37 at the same three levels.

**How large the standard errors might be** [inf, rough]:
- **Lower bound.** With errors independent across the 625,677 rows, SE(γ_P) is at least 2.2/(0.028·791) ≈ 0.10 (0.11 with two-way effects), and SE(κ^c) at least 0.36 to 0.55.
- **More realistically.** The identifying variation of γ_P and κ^c is persistent within cells and the residual is clustered by conglomerate (G* 5.58 in the ext1 fit [V-data]). With the 10,776 bank×market cells (3,581 markets, 69 conglomerates [V-data]) as the effective sample, SE(γ_P) is near 0.8, the size of the parameter.
- **W4.2 should measure this rather than guess it.**

##### 2.5 The launch and the fixed effects

- **The logit has no fixed effects** [V-code: `blp_logit.jl` header 25–29 "full_dtype: δ = α · spread + X · β + γ · dummy_D_type"; 93–94 `X_COLS` without a constant; 143–149: "Deposit-type / quarter fixed effects ... belong to the identification battery, not to this estimator"].
- **So nothing absorbs Post.**
  - **γ_P∅'s regressor** is about 0.43 × Post with a modest cross-sectional spread. In the logit's design it is read off the level of y after 2020Q4. That level also moves with the pandemic, the emergency aid, the start of the tightening cycle, and the sleepy share, through the capped-φ factor of section 1.2 [inf].
  - **With quarter effects,** γ_P∅ rests on the post-launch spread of ν(1 − A^∅): SD 0.078 in the base, 0.055 at 0.999. That spread comes from the market's new-money share and from advertising's effect on A^∅.
  - **γ_P's regressor** is cross-sectional (the rivals' share of W), so quarter effects leave it, but it is small.
  - **The difference γ_P − γ_P∅** is read from the contrast of the two columns across cells. The γ_P side is the weak one.
- **What would add variation for γ_P∅: the dose.** Launch × the market's Pix use, instrumented by the 2019 exposures, gives the new-money column cross-market variation that quarter effects do not absorb [inf].
- **A second, separate reading of γ_P∅ is the opening rate** (eq. 17 uses A^∅). That is W3.2 (openings at the launch, B and D).

##### 2.6 Verdict

**New money is nearly all of the awake pool** in the typical cell under the base cap, and at the cap 0.999 it is the whole of it to two decimals. Own and rival money differ too little for κ^c to be identified at all. γ_P shows only in the top decile of cells, which are large markets with large B stocks, and there it competes with an approximation error of the same size. γ1 is identified, but through new money. γ_P∅ is identified only by the level break at the launch, unless quarter effects and the dose are added.

This is a property of the market size, which is about 30 times the B stocks at the median market-quarter (b_∅ 0.970), and not of the first-pass form [V-data, inf].

---

#### 3. Plan for W4.2: the recovery check (small and local)

**Goal.** Data generated by eq. (12), (13) and (16) at known (γ1, γ_P, γ_P∅, κ^c), fitted with the first-pass form, and with the first-order-corrected form beside it. Measure the bias, and the precision under noise of the data's size.

**Script.** A new `diagnostics/diag_w4_availability_sim.py`, which needs Pedro's go (section 4 of the reply: "Yes, after W4.1"). It reuses K2's context: pools, φ̂, M̂, the menu's e_k, SD. It adds three things:
1. **`pools(ctx, M)`:** as in `w4_feasibility.py`.
2. **`shares_exact(ctx, θ, δ)`:** eq. (13) with μ = 0, source by source, aggregated with the pools.
   - Exact enumeration over the market's B banks where they are 12 or fewer. That covers 99 %+ of market-quarters: 2.9 banks on average, 99th percentile 9 [V-data].
   - 2,000 simulated sets with a fixed seed for the few large markets (up to 52 banks).
   - D institutions always available in v1, a stated simplification; their own availability in v2.
   - **Gate:** with every π = 1, the shares equal the logit's and K2's κ = 0 inflows to 1e-10.
3. **The estimator:** y = ln s^Act_jk − ln s^Act_0 = [α spread + Xβ + γ dD] + ln W(θ) [+ ε(θ)].
   - The linear part is concentrated out by OLS (oracle) or by 2SLS with the logit's instruments (realistic).
   - θ = (γ0, γ∅, γ1, γ_P, γ_P∅, κ^c) by Levenberg-Marquardt.
   - Each fit is done without effects and with quarter effects.

**Truths** [inf: γ1 from the openings elasticity 0.086 at availability 0.5]:
- γ1 = 0.17, γ_P = 1, κ^c = 1, γ0 = γ∅ = 0;
- γ_P∅ = 0 ("cost only"), and a second world with γ_P∅ = 1 ("awareness only").

**Runs.**
- **A. Oracle, full sample, both caps.** δ is known and only θ is fitted, so this isolates the approximation's bias.
  - Cost about 15 to 20 min: shares 2 to 5 min, then 2 truths × 2 forms × 2 caps × about 2 min per fit [inf].
  - Heavy, so the command is reported READY once the script exists.
- **B. Realistic noise, a random 300-market subsample** (seed 20261112).
  - δ_r = the logit's fit + ξ_r, with ξ_r resampled within type from the logit residual.
  - R = 50 replications (seeds [20261112, r]) × 2 truths × 2 forms, about 10 s per fit: about 35 min [inf].
  - It reports, for γ1, γ_P, γ_P∅, γ_P − γ_P∅ and κ^c: bias, Monte Carlo SD, share converged, and the Jacobian's condition number at the truth.

**Expected from section 2** [inf]:
- the oracle recovers γ1 and γ_P∅, and γ_P less well;
- the realistic run gives κ^c a Monte Carlo SD far above 1, and γ_P a Monte Carlo SD near its own size.

---

#### 4. Plan for W4.3: the fit on the data

**Data.**
- The production E3 frames, plus the p4 frames as a variant.
- B rows (625,677) with y = the stored logit δ, the closed form of ln s^Act − ln s^Act_0.
- D rows: W ≡ 1 in v1 (decision W-c).
- Pools from K2's context. The shifters as in W4.1: P2's ln(1 + real advertising) for observed statuses, Post, each source's TD, and γ2 = 0.

**Model:** y_jkmt = α spread + Xβ + γ dD + ln W_jmt(θ) [+ ε_jmt(θ)] + ξ.

**Estimation.**
- θ by Levenberg-Marquardt.
- At each θ, the linear part by 2SLS exactly as `blp_logit.jl` `estimate_theta1_logit`: spread endogenous with its per-type first-stage projection, and the 15 + 1 instruments of the logit.
- Moments for θ: the Jacobian columns of ln W built from the exogenous shifters and the pools.

**Variants:**
1. the logit's design (no effects);
2. with quarter effects;
3. the dose (launch × Pix use, instrumented by the three 2019 exposures), with its own coefficient in each index;
4. advertising instrumented by its four-quarter lag (the P5 instrument);
5. weighted by the own lagged stock, which puts weight where rival money is (stock-weighted rivals' median 0.17 against 0.03 unweighted);
6. cap 0.999.

**Inference.**
- Clustered by conglomerate.
- The score wild cluster bootstrap (Webb weights, 9,999 draws), as `kappa/k1_netflow_nls.py` does, with G* reported.
- For γ_P − γ_P∅, the difference is its own parametrization: index terms (γ_P∅ + d)·Post for rivals and γ_P∅·Post for new money.

**Keys** (proposed prefix `avail.idx`):
- `avail.idx.{g1, gP, gPn, gP_minus_gPn, kc}.{est, se, lo, hi}.{base, qfe, dose, ivlag, wstock, cap0999}`;
- `avail.idx.{n, G, Gstar, converged}.<variant>`.

**Cost** [inf]. About 1 to 3 min per fit locally: 600 thousand rows, 1.1 million pairs, 6 θ, and a 2SLS step per evaluation. The score bootstrap is cheap. Six variants take about 20 to 30 min, so the command is reported READY once the script exists.

**What "shows life" means, fixed in advance** (decision W-d):
- the fit converges in variants 2 and 3;
- γ1 > 0 with a 95 % interval excluding 0;
- the interval of γ_P − γ_P∅ is narrower than ±1 in the index's units.

---

#### 5. Q7.3: does existing code compute the pools' shares b?

**Partly. The inputs exist; the shares do not.**

**What exists:**
- **The accrued stock by source bank-market-quarter.** `diag_k2_netflow_simulation.py` `build` (lines 398–412) computes it: `U["W"]`, types 1, 2, 4 and 5, each accrued at its cell's lagged rate. Also `carry4`, the type-4 part [V-code].
- **The market size** `M_hat` and the outside pool `O` = M̂ − Σ lagged B stocks, by market-quarter:
  - in K2's `MT` (lines 433–447);
  - in K0's `outside_pool` (`diag_k0_netflow_objects.py` 392–405), with `O_over_M` saved to `Rout/k0_outside_pool.csv` [V-code; V-data].
  - K0's O uses unaccrued stocks; eq. (16) uses accrued ones. The difference is the quarter's interest.
- **K0's pool weights `Q1`, `Q4`, `Rk`** are relative to own savings, not to M̂ (lines 237–264) [V-code].
- **The prep** computes M_mt (`sleep_demand_prep_link.py` 424–443, 478) and Dep_Act, with no pools by source [V-code].

**What is new:**
- **The shares b against M̂:** b_jmt = W/M̂, the rivals' total, and b_∅ = 1 − ΣW/M̂. Each is one division (`w4_feasibility.py` does it).
- **The source-specific rival sum** Σ_{j′≠j} b_j′ Ã_{j|j′}. It needs a pairwise sum within each market-quarter, because TD varies by source: 1,113,112 ordered pairs in the a216 context [V-data].

**For Q7.1 and Q7.2, with the formal answers left to the other agent:**
- The logit carries no fixed effects (2.5).
- P2's advertising is national by conglomerate and quarter. It is observed for 99.997 % of destination units [V-data]. The gaps are Crefisa (5 of 36 quarters), Master (9 of 15), VR (30), Social Bank (32 of 35), Pleno and Banco da China (33), Cédula and Cargill (34) [V-data: `Rout/p2_note.md`].

---

#### 6. Decisions for Pedro, and IMPR's recommendation

| # | decision | options | IMPR recommends |
|---|---|---|---|
| W-a | order | W4.2 first, then W4.3; or W4.3 directly | **W4.2 first:** about 1 h locally, and it says whether γ_P and κ^c can be recovered at all |
| W-b | the estimating form | first-pass only; plus the first-order correction ε(θ) | **both:** the error is the size of γ_P's identifying variation |
| W-c | D rows | W ≡ 1 (outside the index); national pools | **W ≡ 1** in the first pass. D-held money stays in new money, a stated simplification |
| W-d | "shows life" | the criterion of section 4, fixed before the run | yes |
| W-e | what to report if W4.2 confirms section 2 | report γ1 and γ_P∅ (with quarter effects and the dose), and state γ_P − γ_P∅ and κ^c as not identified in this form; or go to the full share system; or revisit the market size | **the first:** the full system has the same pools, so the same weakness [inf]. Revisiting M̂ is a model decision for Pedro, since the cap 0.999 makes it worse (anchor ×5) |
| W-f | the trend in b | as is; deflate M̂ to grow with deposits | Pedro's call. Note that one anchor for all quarters makes the rival share rise mechanically after 2020 (2.1) |

**IMPR's recommended option.**
1. Write `diag_w4_availability_sim.py` (Pedro's go) and run W4.2 A and B locally, about 1 h, one process.
2. Then W4.3 with variants 2 to 5 at the base cap.
3. Expect, and write in advance, that the first pass identifies advertising and, with the dose, Pix's effect on new money. Pix's switching-cost part and the clock rest on cells too few to carry them.

---

#### 7. To reproduce

All commands are read-only and take under a minute; they write only into `scratchpad\w4_1112\`:

`C:/venvs/egan/Scripts/python.exe -I w4_feasibility.py "<scratchpad>/kappa_1109/k2n_ctx_fa38794d_a216.pkl" "C:/Users/pedro/AppData/Local/Temp/capsens_1111/p4/demand_prep/demand_3_index_spec_12.parquet"`

`C:/venvs/egan/Scripts/python.exe -I w4_adv_margins.py "<Rout>/p2_advertising_conglomerate_quarter.csv"`

`C:/venvs/egan/Scripts/python.exe -I w4_bnew_by_year.py "<scratchpad>/kappa_1109/k2n_ctx_fa38794d_a216.pkl"`

## Update 11-12 16:20: W1.1, W1.2 and Q7.1 done: the pooled α rests on checking and savings, and the instrumented types alone give a positive α

All of it is read-only.
- Scripts and JSON outputs are in `scratchpad\w1_1112\`, each running in under a minute.
- The logit is reproduced exactly: pooled α −0.064079813.
- Production frames, full_dtype, spec 12, conglomerate clusters throughout.

### W1.1 (b): what identifies the pooled α (Pedro's point holds)

| sample | E3 α (WCB se) | p | n | G / G\* | E4 α (se) |
|---|---|---|---|---|---|
| pooled (production) | −0.0641 (0.0315) | 0.092 | 628,612 | 300 / 5.6 | −0.0637 (0.0321) |
| **types 1–2, OLS, spread exogenous** | **−0.1185 (0.0118)** | < 0.001 | 384,245 | 107 / 5.7 | −0.1216 (0.0119) |
| **types 4–5, instrumented as in the logit** | **+0.893 (0.321)** | 0.036 | 244,367 | 294 / 5.3 | +0.929 (0.312) |
| type 4 alone, instrumented | +0.355 (0.170) | 0.086 | 181,326 | 163 / 5.4 | +0.412 (0.186) |
| type 5 alone, instrumented | +2.80 (1.50) | 0.13 | 63,041 | 181 / 4.4 | +2.81 (1.46) |

- **The pooled α rests on the checking and savings spreads.** As a weighted average of the two groups, types 1–2 carry 106 % of the identifying variation and types 4–5 enter with weight −6 %.
- **The instrumented types alone give the wrong sign.**
- Types 1–2 alone give −0.119 rather than −0.064 because the pooled regression forces common coefficients on the characteristics [inf].

### W1.1 (a): Anderson–Rubin sets for the pooled α (E3; E4 is close)
Two statistics: a cluster-robust Wald with a χ² reference, and a fixed-weight statistic with a restricted wild-cluster null (Webb weights, 999 draws).

| instruments | L | Wald set | bootstrap set | t(G\*) set |
|---|---|---|---|---|
| all the logit's excluded instruments | 33 | **empty** (min 37,076 against 47.4) | **empty** (max p 0.012) | — |
| Z for types 4 and 5 only | 32 | empty | empty (E4: [0.12, 0.29]) | — |
| **the checking/savings spread only (exogenous)** | 1 | [−0.139, 0.011] | [−0.171, 0.038] | **[−0.159, 0.031]** |
| the logit's own generated instrument (not weak-IV robust) | 1 | [−0.127, −0.002] | [−0.155, 0.022] | [−0.144, 0.016] |

- **Under the logit's full set of exclusions, no α satisfies all the moment conditions.** The joint restrictions are rejected, so no robust interval exists under them.
- **Under the exogeneity of the checking and savings spreads alone,** the robust interval is **[−0.159, 0.031]**. It contains zero and matches the stored p of 0.092.

### W1.1 (c): Montiel Olea–Pflueger critical values (5 % level)

| first stage | F_eff | K_eff | τ = 5 % | τ = 10 % | τ = 20 % | τ = 30 % |
|---|---|---|---|---|---|---|
| type 4, E3 | **1.36** | 2.05 | 32.2 | 19.2 | 12.0 | 9.4 |
| type 4, E4 | 1.34 | 2.04 | 32.2 | 19.2 | 12.1 | 9.4 |
| type 5, E3 | 2.22 | 1.71 | 33.4 | 20.0 | 12.7 | 10.0 |

- **The method** is the simplified, conservative TSLS procedure, using the clustered Ŵ₂.
- **A check:** with K_eff = 1 it reproduces MOP's published 37.42, 23.11, 15.06 and 12.05.
- **The verdict:** every MOP critical value with at most 16 instruments is at least χ²₀.₉₅(16)/16 = 1.64. So **type 4 is weak at every τ under either procedure.**

### W1.1 (d): the other first stages

| first stage (where) | instruments | Wald F | F_eff | 10 % / 20 % critical value | G / G\* |
|---|---|---|---|---|---|
| BLP ext1 linear IV (`blp_engine_cpu.jl:2141`, the same projection as the logit) | 16 | 22.3 / 0.42 (types 4 / 5) | **1.36 / 2.22** | 19.2/12.0; 20.0/12.7 | 163/5.4; 181/4.4 |
| **sleepiness control function**, spec 12 (`sleep_est_e2.run_pooled_first_stage`, from the est3/est4 pickles; E4 identical) | 7 | 13.07 | **4.96** | 20.5 / 13.1 | 446 / 6.2 |
| stakes, P3 (one instrument) | 1 | 781.8 (571.8 net of market structure) | 781.8 | 23.1 / 15.1 | 2,288 / 2,115 MCAs |
| Pix use, K1's U-hat (P1) | 2 | 65.3 HC1 / 70.5 states | 42.7 / 40.0 | 21.2 / 13.6 | 3,333 MCAs; 27 / 12.3 states |
| Pix exposure E1 (3 exposures; MCA / state clusters) | 3 | tx 44.3/79.6; payers 33.5/43.7; value 10.8/5.9 | tx 29.9/54.4; payers 24.8/21.6; **value 11.4/8.1** | about 20.6 / 13.1 | 3,333; 27 / 12.3 |

- **Weak:** the type-4 and type-5 demand first stages, also in BLP's linear step, and the **sleepiness control function** (F_eff 4.96 against a 20 % critical value of 13.1).
- **Strong:** the stakes, Pix use, and the transactions and payers exposures. The value exposure is borderline.
- The `sleep.fs.*` keys are absent: `sleep_first_stage_strength.json` waits for the weak-IV battery, deleted on 10-16. The control-function numbers above are computed from the pickles.

**The three directions that hold 86 % of the type-4 first-stage trace are conglomerate-level cost and capital instruments:**

| share of the trace | main instruments (correlation) | conglomerates driving it |
|---|---|---|
| 50 % | Basel index +0.54, personnel cost +0.45, tax cost −0.34 | Santander 34 %, Bradesco 24 %, BB 24 % |
| 20 % | tax cost −0.58, administrative cost −0.22, mean LOO equity +0.22 | Itaú 60 %, BB 30 % |
| 16 % | Basel index −0.77, tax cost −0.25 | BB 27 %, Bradesco 26 %, State Street 13 % |

- For type 5, the cost instruments dominate, driven by Cora SCFI (66 %), C6, Stone IP and Shopee IP.

### Q7.1: the logit's fixed effects: NONE (IMPR verified)
- **The design:** no intercept (`BLP_LOGIT_INTERCEPT` off), and no quarter, bank, market or deposit-type effects.
  - `blp_logit.jl` l.93–94 and l.144–149: the published logit is the θ₂ = 0 nest of the RC model, whose X₁ has no constant. "Deposit-type / quarter fixed effects … belong to the identification battery, not to this estimator."
  - The BLP engine's linear design is the same.
- **The regressors:** `fgc_covered` (1 for types 1, 2 and 4, so it acts as a group intercept), `has_ip`, `seg_S2`–`S5`, `log_total_assets_lag`, `is_state_owned`, and the D dummy.
- **For the paper:** if the launch shift and advertising enter the logit as it stands,
  - **the launch is identified from the aggregate time-series break at 2020Q4, with nothing absorbing common shocks;**
  - advertising is identified from both cross-bank and over-time variation.
- The sleepiness step, by contrast, carries entity and quarter effects.

### W1.2: what switching production to cap 0.999 (p4), or to p3a, would cost
- **What moves under p4:**
  - the demand frames, E1–E4;
  - the logit, E1–E4: its tables, the comparison table, the δ starts;
  - **RC-BLP ext1, E1–E4,** since Table 6 reports all four: `demand.t6.*`, `demand.elast.blp.*`, ξ̂ and `demand.xi.*`;
  - the kappa keys: K2/K2b, K0's outside pool, and the pending `kappa.gap.*`;
  - the cap keys; TR5;
  - CF/BBL (V_Full only, off now; their φ clamp is already 0.999).
- **What does not move:** the sleepiness estimates and φ̂ (all `sleep.*`), the BLP draws, `trunc_share`, the first stages, and A.2/A.3.
- **p3a adds a different row set** (E3 loses 54 % of B rows), a new production knob (Pedro's go), rebuilt draws, and a CF over only 36 % of markets, so a different object.
- **Jobs under p4:**
  1. the code edit (PHI_CAP default, `sleep_demand_prep_link.py` l.104);
  2. upload;
  3. the sleep phase with `--skip-est` (exports and prep E1–E4; Bouchet, about 1 h);
  4. the logit E1–E4 (about 10 min);
  5. BLP ext1 E1–E4 in parallel (0.5–1.5 GPU-h each);
  6. ingest, exhibits and keys (1–2 h).
  - **The critical path is about 3–5 h,** plus about a day if TR5 and K2b are refreshed.

### Decisions for Pedro (IMPR's recommendation in brackets)
1. **Which interval to report for α:** [the checking/savings-exogenous AR interval [−0.159, 0.031], stating that the full-instrument set is empty because the exclusions fail jointly].
2. **Section 6's account of identification:** [say that α rests on the checking and savings spreads, and that the instrumented types alone give a positive coefficient].
3. **The sleepiness control function is weak** (F_eff 4.96): [state it with the critical values; the deleted weak-IV battery is the place to rebuild its keys].
4. **p4 or p3a:** [p4 only, if any; it costs about 3–5 h of wall time. p3a is a different object].
5. **The `demand.fs.efff.*` keys:** the effective F by first stage, the critical values and the AR intervals. Pedro's go covers new keys; IMPR adds them after the keys agent finishes.

## Update 11-12 16:15: W3 (awareness, reduced form) and Q7.2 done

All of it is read-only.
- Scratch: `scratchpad\w3_1112\` (`aw1_launch_bd.py`, `own_closings.py`, `q72_w33.py`).
- No repo file was edited, and nothing was written to Rout.
- The 11-11 difference test (`pix.adv.diff*`) is as reported on 11-11: six gates pass, three entries at status ok.

### W3.2: openings at the launch, B and D separately (your revised W3.2)
**The specification is AW1's own** (`make_ccs_openings_pix.py`, `B_` and `D_conglomerate_trends`):
- ln openings on post × on-Pix and post;
- conglomerate effects, calendar-month effects, and one linear trend per conglomerate;
- monthly, 2018-01 to 2022-12.
- AW1's estimates are reproduced exactly, and its bootstrap replays (p 0.4685 against the stored 0.468).

| | estimate | wild bootstrap p [95 %] | CR2 p (df) | placebo months with a jump at least as large |
|---|---|---|---|---|
| **B, jump at on-Pix banks** | **+0.177** | **0.427** [−0.238, 0.593] | 0.412 (43.3) | 10 of 22 (rank p 0.48) |
| B, on-Pix minus off-Pix | +0.153 | 0.468 | 0.446 (29.0) | |
| B, banks not on Pix | +0.024 | | 0.894 (30.1) | |
| **D, jump at on-Pix institutions** | **+0.317** | **0.112** [−0.061, 0.703] | 0.113 (80.1) | **0 of 22 (rank p 0.043)** |
| D, on-Pix minus off-Pix | +0.010 | 0.962 | 0.961 (69.0) | |
| D, institutions not on Pix | +0.307 | | **0.024** (80.3) [0.042, 0.573] | |

- **The placebo months** are the same fit on 2018-01 to 2020-10, with a common jump at each month from 2018-07 to 2020-04. They are the check on the single national date, which conglomerate clustering cannot see.
- **There is no few-cluster problem here:** 30–81 effective clusters.
- **Without the trends** (AW1's earlier specification) the jumps are B +0.845 and D +1.92, but AW1's event study shows pre-launch trends. So the trend specification is the reading.
- **Reading:**
  - **B openings did not jump at the launch.** The interval still reaches +0.59.
  - **D openings rose by about 0.3 at the launch,** larger than every placebo month. **But D institutions not on Pix rose by the same amount (+0.307),** so the rise is not specific to Pix participation.
  - In the revised model γ_P∅ is common to all institutions. So the D rise reads as γ_P∅ > 0 for D, unless it is the pandemic, aid or tightening shock that shares the date.

### W3.1: the exclusion on own money
- **P8 has no object for own money.**
  - P4's dependent variable is openings / (adults − N_{j,t−1}): entry of adults who did not hold j, the rivals' and new side. It counts people, not money; own advertising +0.086 [0.026, 0.153].
  - P5 adds the 4-quarter-lag instrument and the placebo on rivals' advertising (−0.79, p 0.002).
  - Nothing touches closings, deposits or markets.
- **The money version needs the pool shares b_jmt** (W4.1 and Q7.3), then an advertising term ψ in δ beside γ1, with ψ = 0 as the test.
  - With new money at about 0.97 of the market size, own and rival money differ little, so its power is likely low. The W4.1 table decides whether it is worth running.
- **A relationship-margin version, computed now:** the PF closing rate (closings / N_{j,t−1}) on own advertising, in P4's design. A holder always considers her own bank, so under the exclusion her closing should not respond to its advertising.
  - **The main sample** (n 1,337, G 46): **−0.008**, bootstrap 95 % [−0.041, 0.024], p 0.581; CR2 p 0.597.
  - **All B** (n 1,743, G 68): −0.005, p 0.743.
  - **Reading:** consistent with the exclusion. It rules out a fall in closings larger than about 0.04 per log unit, against +0.086 for openings.
  - **Caveats:**
    - churn of accounts opened by advertising would push the estimate up;
    - closings include closures the bank starts;
    - N mixes PF and PJ.

### W3.3: `pix.adv.*` are current on I2
- **Panel:** fa38794d.
- **The sidecar's recorded md5s match the inputs:** P4 CSV `112c7975`, P4 JSON `f881a6d5`, P2 coverage `a6ca0b18`.
- **The 23:36 rebuild** has all five `pix.adv*` entries at ok, with 0 MISMATCH.
- **One caveat.** P2's advertising files carry panel d01b8bae, because P2 was not rerun in P8.
  - The amounts do not depend on the panel, and the five-largest label is unaffected.
  - P2's conglomerate-quarter frame is the older panel's. Rerunning `diag_p_shifters.py --only p2`, then P4, would make it current; the effect is expected to be small.

### Q7.2: advertising coverage
- **The source:** the Cosif account "Propaganda e publicidade" (81745009), released on request (LAI 18810.023084) for 2013–2024.
  - It is an accrued expense, summed over a conglomerate's members.
  - **It is national, one value per conglomerate and quarter**; every quarter used has all three months.
  - It is in R$ of 2024Q4, with negative entries set to 0.
- **B, 2016Q1–2024Q4:**
  - observed in 2,220 of 2,277 conglomerate-quarters, **covering 99.96 % of B deposits** (the lowest quarter, 2024Q3, is 99.73 %);
  - 60 of 69 B conglomerates are observed in every quarter they are in the panel.
  - **The gaps are small banks only:** Crefisa (none), Bank of China (2024), Pleno (2024), Master Múltiplo (2023–24), VR, Cédula and Social Bank (2024), Cargill and Intesa (2022).
- **D:** 88 % of D deposits in 2016, rising to 97–99 % from 2018.
  - About 30 % of D conglomerates have no row: payment institutions and credit fintechs (SCDs), IBM, and PicPay before 2023.

### Proposed keys and decisions
- **Keys,** built from a new light diagnostic (about 3 min) with a sidecar, plus a collector edit. This waits for the keys agent now editing `make_paper_numbers.py`, and for Pedro's confirmation:
  - `pix.aw1.jump.B` 0.177 [−0.238, 0.593], p 0.427, `p_cr2` 0.412, `p_placebo` 0.455, with `.off` and `.diff`;
  - `pix.aw1.jump.D` 0.317 [−0.061, 0.703], p 0.112, `p_placebo` 0.000, with `.off` 0.307 (CR2 p 0.024) and `.diff` 0.010;
  - `pix.adv.own.close` −0.008 [−0.041, 0.024].
- **For you and Pedro:**
  1. **What to quote for W3.2.** IMPR suggests B as "no jump", with its interval and the placebo share. For D, the on-minus-off difference (≈ 0), stated next to the common rise.
  2. **The relationship margin** as the first evidence on own money? And should the money version (ψ in δ) run inside W4.3, after the W4.1 table?
  3. **Rerun P2, then P4,** on fa38794d, to make the advertising frame current. It is light and local.

## Update 11-12 15:50: D0 read (E3 and E4); the step limit r0 = 0.02 holds; the validation fits are next

**The jobs.** 33145172 (E3, 1 h 40 min) and 33145175 (E4, 1 h 39 min), both COMPLETED with exit 0.
- They used the production engine and were read-only.
- The production ext1 E3 files hash `7aeb4113…` (checkpoint) and `abf1ecb5…` (results). The comparison with D0's recorded hashes is being confirmed.

**(ii) The first-step scan.** The scan's t is a fraction of L-BFGS-B's full first step (the unscaled projected gradient). Converted to the move in θ's largest coordinate:

| start (E3; E4 is the same) | full first step | the inner contraction still converges at | it fails at | Q along the step |
|---|---|---|---|---|
| cold | 2.46 | 0.25 | 0.74 | falls to 0.025 (−0.040), then rises by 0.074 (+0.017) |
| rand1 | 7.49 | 0.75 | 2.2 | **rises at every t:** the direction is wrong |
| rand2 | 6.48 | 0.19 | 0.65 | falls throughout (−30.8 at 0.19) |
| rand3 | 6.89 | 0.69 | 2.07 | falls throughout |

**The reading.**
1. **lite's r0 = 0.02 holds.** The first trial moves by 0.02, at least 10 times inside the radius where every start's contraction still converges (0.19 or more). For cold, which overshoots by 0.074, 0.02 is a descent step. **The validation runs as prepared.**
2. **rand1's first direction is wrong** (dQ > 0 at every t). Its contraction at θ₀ failed at 5,000 iterations, so the gradient came from an unconverged δ (|∇Q| 8.7e5 for E3, 4.3e6 for E4). From a clean δ the same contraction converges in 4,584 (E3) and 4,908 (E4) iterations. **b3's 10× first-evaluation cap covers exactly this.**
3. **A polluted δ moves Q even after the contraction converges.**
   - Re-evaluating θ\* from the δ that rand2's (E3) or rand3's (E4) first trial left gives a Q higher by **2.6e-4** (E3) and **4.5e-4** (E4) once converged.
   - That is the same order as the differences between starts. **b3's restart from the last converged δ addresses this.**
4. **The logit δ̂ start.**
   - E3 converges in 3,684 iterations; Q is 2.9e-7 below production.
   - **E4 fails at production's 5,000 cap** (dQ −5.2e-4). It converges in 3,127 iterations on re-evaluation, with dQ −1.7e-6.
   - This matches the frozen E4 warm start of 11-06.
5. **Where the contractions fail.**
   - At far starts, every observation on the plateau sits at |ln 0.95| in markets at the inside-share floor: 363 of 363, 37/37, 79/79 and 527/529 markets.
   - There the model's outside share is about 1e-5 to 1e-4, against the data's 0.05.
   - At θ\* from the logit δ̂, after 300 iterations, 255 (E3) and 391 (E4) observations sit there, all in floor markets.
   - **The floor markets (3.6 % of market-quarters) are where the BLP contraction stalls.** This is relevant to N6 and to s3a/p3a.
6. **Sanity checks.**
   - From δ\*, the re-evaluation reproduces production's Q (dQ −2.4e-17 for E3, −3.7e-16 for E4).
   - μ is identical across the ext2 and extended stages.
   - (i): at θ₀, cold converges, rand2 converges in about 3,100 iterations, and rand1 and rand3 fail. Every unscaled first trial fails, with 9–100 singular blocks and 585–5,593 δ at ±500.

**Done on Bouchet at about 15:55:**
- **D0's recorded hashes equal production's** (`7aeb4113…`, `abf1ecb5…`), so D0 left production untouched.
- **The engine is PROMOTED:** production `scripts/` now holds `7d34eaa7` / `c5b9c692`, with its flags off by default.
  - The backup of `2b467eb6` / `16261cde` is in `~/scratch_pi_mf2263/pf382/engine_prod_backup_20261112/`.
- **The 12 validation sandboxes are built (`val1112`):**
  - the controls start from ext1's δ\* (E3 `80bd9796…`, E4 `f7ecf3b1…`);
  - warm_rc4 starts from rc4's;
  - cold and rand1–3 take no stored δ.
- **The engine is in all 12 sandboxes** (12 × `7d34eaa7`, 12 × `c5b9c692`).
- **The 12 validation fits were submitted at about 16:00:**
  - E3 jobs 33171617–33171622, E4 jobs 33171623–33171628;
  - flags: lite with r0 0.02, b3 on, b4 on; wall 1 day.
  - **Starts:**
    - the controls at ext1's θ\* (with δ\*);
    - warm_rc4 at rc4's solution (with rc4's δ);
    - cold;
    - rand1–3 at production's random points.
  - At most 6 run at once, so two waves of about 6–12 h.
  - **The flags reached the jobs** (Pedro's check, about 17:05): 10 started jobs log "[OUTER] stop reason on | step lite (r0 0.02) | inner guard on | clamp test on". The other 2 had not yet printed the line.
- **Pass:**
  - control: Q within 1e-9 of production, and FD passes;
  - warm_rc4: ends at θ\*;
  - cold and rand1–3: move, and pass FD in at least 3 of 4 per routine;
  - production's ext1 files unchanged.

**Next on Bouchet** (Pedro's go, per IMPR's recommendations):
- the hash confirmation;
- the engine promotion (backup, then copy);
- **the 12 validation fits:** build, put the engine into the sandboxes, then submit with lite r0 0.02 + b3 + b4, a 1-day wall, about 12–30 GPU-h and 6–12 h of wall time.

## Update 11-12 15:40: Pedro's go on four items (about 15:30)

**1. The IF.data second vintage is filed.**
- **Where:** `DATA_RAW\BCB\segmentacao_prudencial\ifdata_cadastro_refetch_2026-11-12\`.
  - 36 files, 32,645,621 bytes, every copy's md5 verified against the download.
  - `MD5SUMS.txt`, manifest md5 `595c9086…`.
- **NOTES.md** (`f095cc5b…`) has a new section, "Second vintage", with the differences and the 0.054 % effect.
- **One row appended** (append-only, prefix byte-identical) to:
  - the master `source_log.csv` (414 → 415 lines; md5 `e4d6bf9a…`);
  - the folder's `source_log_rows.csv` (`3d9bd726…`).
- The segment map keeps the archived 04-07 copy.

**2. The new keys and the `tab_tax_clock` Panel B header: agent running.**
- **Keys:** `kappa.gap.*` (Part A), `demand.capsens.*` (the cap screen, with p5 and p6 admissibility), and `demand.findex.*` (constant 0.77; series 2016 and 2024).
  - Each comes with sidecar and md5 guards. The rebuild must keep every existing value unchanged.
  - `demand.fs.efff.*` follows W1.1.
- **The header:** only the header line of the generated table may change, via `make_tax_clock_table.py`.

**3. The engine promotion: lines given to Pedro**, to run once D0 has finished:
- a backup of production's engine (`2b467eb6` / `16261cde`);
- a copy of the staged `7d34eaa7` / `c5b9c692` into `scripts/`;
- md5 checks, and a rollback line.
- The flags stay off. R0 showed that the defaults reproduce 11-06 exactly.

## Update 11-12 15:35: `MAIN_REPLY_2026-11-12.md` read; the no-go items started

All of it is read-only. The heavy jobs are still one at a time; agents report READY first.

| item | who | content |
|---|---|---|
| **W1.1** weak IV on the pooled system | agent (the cap-screen agent, resumed) | the cluster-robust AR interval for α; types 1 and 2 without an instrument against types 4 and 5 instrumented; the MOP critical values for the type-4 first stage; the effective F and effective clusters of BLP ext1's linear IV (types 4, 5), the sleepiness control function, the stakes and Pix use; which instruments carry the three dominant directions |
| **W1.2** the cost of a looser cap | same agent | for p4 (cap 0.999) and p3a: every output that moves, every job to rerun, and the time |
| **Q7.1** the logit's fixed effects | same agent | from `blp_logit.jl` |
| **W3** awareness, reduced form | agent (the advertising-test agent, resumed) | the own-money exclusion test: what P8 has and what is missing; the opening margin at the launch (AW1) as a number with an interval; `pix.adv.*` on I2 |
| **Q7.2** advertising by conglomerate | same agent | national by conglomerate in every quarter? which quarters are missing? |
| **W4.1** feasibility of the availability index | agent (the K2-plans agent, resumed) | the distributions of b_jmt, the rivals' total and b_∅mt (base and cap 0.999); how much of the variance of ln W each shifter can move; the variance of advertising and TD; whether the fixed effects absorb the launch indicator. It also checks your first-pass algebra against the exact consideration-set share, and plans W4.2 and W4.3 |
| **Q7.3** the pools' shares b in existing code | same agent | K0, K2 and the prep |
| **W2** plan: the stakes in the sleepiness step | new plan agent | files and lines; the instrument; the placebo; local or Bouchet; the keys |

**Waiting:**
- W1.3 (BLP ext1 for p4 and p3a) waits for D0, which is still running on Bouchet.
- **The new keys of your section 6** (`kappa.gap.*`, `demand.capsens.*`, `demand.findex.*`, and later `demand.fs.efff.*`), the engine promotion, and the IF.data second vintage wait for Pedro's go, which IMPR has asked for.
- **After Pedro applies block S7:** IMPR reruns `make_paper_equations.py`.
- **The parked K1 and K2 items:** noted. No keys are deleted.

## Update 11-12 15:25: R0 passes on Bouchet: the edited engine at default flags reproduces 11-06 exactly

- **The job:** 33145739 (`ext2s_E4_x1zero`), COMPLETED, exit 0:0, 1 h 03 min. It re-ran the 11-06 E4 ext2 x1zero fit in a fresh sandbox with the staged engine (`7d34eaa7` / `c5b9c692`) and no `BLP_*` flags.
- **The result:**

  | | the 11-06 run (old engine) | R0 (edited engine, defaults) |
  |---|---|---|
  | Q | 0.035847297135117 | **0.035847297135117** |
  | θ₂ | [0.0, 0.9124016, 0.27711917, −0.015627417, −0.10742462, −0.81616749] | **identical to every printed digit** |

- **The new `--report` columns work:**
  - stop `CONV:F` (the relative reduction of Q), 22 evaluations;
  - FD pass (best dQ +9.9e-08), moved 0.82;
  - the step control off and the guard off, as intended at the defaults.
  - "production ext2 files unchanged? yes: every hash matches".
- **Not to be run:** the report's suggested `blp_ext1_adopt.sh 4 x1zero`. R0 is a reproduction check, and its Q equals production's to 1e-10.
- **What R0 settles.** On the GPU, the edit (b1 on by default; b2–b4 off) leaves today's results unchanged. Per IMPR's recommendation, which Pedro accepted, the edited engine can be promoted into production `scripts/` with its flags off. The two lines are ready, a copy and an md5 check, and wait for Pedro's moment.
- **What it does not settle:** whether b2–b4 fix the frozen starts. That is the validation fits' job, after D0.

## Update 11-12 15:25 (b): K1 300-market R-q pilots done: κ̃_td is interior, but nothing converged

**The design.** These are the S-stmt pilots' exact design with `--lambda R-q` (t−1 timing, fill B):
- n 16,701, G 14 (G\* 5.28), 142 markets;
- starts 0, 0.5, 1 and 1.5, maxit 500.

**The runs:**
- headline 321 s and savings-only 307 s, both rc 0;
- script `8920a40a`, unchanged;
- outputs in `scratchpad\k1_lambda_1111\pilot300_Rq_*`; nothing written to Rout.

**κ̃_td does not hit its box at 300 markets.**
- The reported solutions give **0.323** (headline) and **1.312** (savings-only).
- At 50 markets it sat at the upper box of 10.
- Only one start hit any box (headline start 0: κ̃_chk and dκ̃_chk), and **both reported solutions are interior.**

**But nothing converged.**
- Every start ended on "no descent step" or maxit, as the S-stmt pilots did.
- The standard errors are very wide. These are pilots under K1's standing caveat.

**Side by side with the S-stmt pilots** (same rows, same y and weights, so SSR/yy is comparable):

| form | λ | solution | SSR/yy | dκ̃ (se) | κ̃_td (se) | α (se) |
|---|---|---|---|---|---|---|
| headline | S-stmt | interior (start 0) | 0.2181 | −0.788 (6.13) | not estimable | −0.020 (0.033) |
| headline | S-stmt | at a box (0.5) | 0.1898 | −5.244 (7,606) | not estimable | −0.148 (0.029) |
| headline | **R-q** | **interior (1.5)** | **0.1869** | −2.679 (18.3) | **0.323 (6.36)** | −0.0004 (0.031) |
| savings-only | S-stmt | interior (start 1) | 0.2198 | −0.341 (2.11) | not estimable | −0.003 (0.043) |
| savings-only | S-stmt | at a box (0) | 0.2174 | +0.303 (0.69) | not estimable | −0.227 (0.034) |
| savings-only | **R-q** | **interior (0)** | **0.1882** | −4.854 (53.4) | **1.312 (2.19)** | −0.038 (0.041) |

**Reading.**
- **R-q fits the net flows better:** SSR/yy is about 0.03 lower than S-stmt's interior solutions. Part of that is mechanical, since R-q has two more parameters (κ̃_td and dκ̃_td).
- **Its best solutions are interior, where S-stmt's best were at a box.**
- **dκ̃ is negative in both forms, with very wide intervals:**
  - headline: WCB [−39.9, 34.5];
  - savings-only: [−93.1, 83.4].
  - So the R versions do not tighten the bound on dκ̃.
- **The savings-only time-deposit change dκ̃_td is −6.70** (se 4.69, WCB [−14.7, 1.3]).
- **Score tests:** dκ̃ = 0 gives p 0.815 (headline) and 0.284 (savings-only).
- **For the text:** option (iii) stays. The R version estimates κ̃_td and ln E_c without hitting a box, but no fit converges, and dκ̃ is no better identified than under the statements' λ.

**Decision for Pedro:** whether to promote these R-q pilots into Rout and keys (`kappa.k1.pilot.Rq.*`).
- IMPR's recommendation: only if the text will report the R version next to the statements' version. Otherwise keep them as a recorded robustness check in this reply.

## Update 11-12 14:55: cap sensitivity (Part B, Stage 1) done; the logit's effective F verified at 1.36; K1 300-market pilot started

**The run.** 12 local logits, E3 and E4, 14:07–14:47.
- Every variant exited 0. Each log's s₀ median equals the closed form to 4 dp.
- **Gate B-S3 passes:** |α(base\*) − α(production)| is 5.3e-7 (E3) and 7.3e-6 (E4), against bounds of 0.0016. So re-sharing the stored rows reproduces production, and **no variant needs Stage 2.**
- **Files:** `Rout/diag_cap_sensitivity/` (`cap_screen_summary.csv` `3a597d46`, `.json` `3f21428c`, `MD5SUMS.txt`).
- **Script:** `diagnostics/diag_cap_sensitivity.py` (`bfd2124d`, new).
- **Sandbox:** frames and logit outputs, 1.3 GB, in `C:\Users\pedro\AppData\Local\Temp\capsens_1111\`, removable when done.
- **Spot-checked by IMPR** against the CSV.

**E3** (the full_dtype logit; WCB se; elasticity = α × mean ρ(1 − s); E4 in the CSV, with the same pattern):

| variant | caps (φ / share) | α (se) | p | elasticity | median outside share | n |
|---|---|---|---|---|---|---|
| production | 0.99 / 0.95 | **−0.0641** (0.0315) | 0.092 | −0.245 | 0.864 | 628,612 |
| **f, Findex by year** | 0.99 / 0.95 | **−0.0653** (0.0313) | 0.086 | −0.249 | 0.859 | 628,612 |
| p1 | 0.98 | −0.0484 (0.0312) | 0.175 | −0.185 | 0.872 | 628,612 |
| p2 | 0.995 | −0.0770 (0.0316) | 0.054 | −0.295 | 0.878 | 628,612 |
| p4 | 0.999 | −0.0923 (0.0314) | 0.028 | −0.354 | 0.938 | 628,612 |
| p5 | 0.9999 | −0.1052 (0.0310) | 0.016 | −0.405 | 0.980 | 628,612 |
| p6 | 0.99999 | −0.1181 (0.0309) | 0.010 | −0.456 | 0.997 | 628,612 |
| s1 | share 0.90 | −0.0608 (0.0313) | 0.104 | −0.232 | 0.864 | 628,612 |
| s2 | share 0.98 | −0.0679 (0.0317) | 0.079 | −0.259 | 0.864 | 628,612 |
| p3a: drop B rows at the φ cap | 0.99 | −0.0826 (0.0326) | 0.045 | −0.363 | 0.835 | 289,773 |
| p3b: drop them before r̂ | 0.99 | −0.0780 (0.0326) | 0.055 | −0.342 | 0.805 | 289,773 |
| s3a: drop floor market-quarters | 0.99 | −0.0457 (0.0305) | 0.189 | −0.179 | 0.873 | 587,644 |

**Reading.**
- **The φ cap moves α steadily.** α is −0.048 at 0.98, −0.064 at 0.99, −0.077 at 0.995, −0.092 at 0.999, and −0.118 at 0.99999. E4 is the same: −0.051 / −0.064 / −0.073 / −0.086 / −0.119.
  - The 5 % verdict changes from p4 on.
  - p5 and p6 fail admissibility (M0): ordinary markets' inside share is 0.0036 and 0.0005, so ordinary markets have an outside share of 98–99.7 %.
- **The share cap hardly matters** (s1, s2).
- **The Findex series moves α by 0.04 se (f)**, and the outside share by −0.005. So **the Findex fix (N7) is immaterial for the logit.**
  - It can wait for the next rebuild, with the text corrected now: %AE6, %AE7, %AD7.
- **Dropping the rows at the φ cap (p3a)** gives −0.083: p 0.045 in E3, 0.061 in E4. Part of its elasticity move is a change of sample (mean ρ(1 − s) 4.39 against 3.82).
- **Excluding the floor markets (s3a)** gives −0.046 (p 0.19).
- **The proposed BLP-trigger rule** (B.6, not approved) is met by p1, p2, p4, p3a, p3b and s3a; p5 and p6 fail M0.
  - Its M1 threshold (a 10 % elasticity move) is crossed at about 0.4 se, so it triggers easily.
  - **Whether to run BLP ext1 for any of them is N6, Pedro's.** IMPR would run only p4 (the cap) and p3a (the sample), if any.

**The logit's type-4 first stage, now VERIFIED as a computation.**

| | E3 | E4 |
|---|---|---|
| clustered Wald F | 22.3 | 25.0 |
| **Montiel Olea–Pflueger effective F** | **1.36** | **1.34** |
| effective F without the rows at the φ cap | 2.28 | 2.13 |
| effective F without the floor markets | 1.56 | 1.58 |

- **The cross-checks:**
  - statsmodels' cluster-robust OLS, an independent library, gives the same Wald F, 22.2717.
  - F_eff from statsmodels' covariance is 1.354594955362, against 1.354594955361.
  - Montiel Olea and Pflueger's normalised formula, coded from scratch, gives 1.363; the difference is the small-sample factor of 1.0063.
  - The textbook case holds: with one instrument, F_eff equals the robust Wald F to 2e-16.
  - Type 5 gives F_eff 2.22.
- **Why the two statistics differ.**
  - The clustered covariance of the first-stage coefficients is nearly singular: its eigenvalue ratio is about 7,000, three directions hold 86 % of its trace, and there are about 5.4 effective clusters.
  - The Wald F inverts that matrix and is inflated by it; the effective F uses its trace.
- **Not done:**
  - the critical values, so there is no formal verdict yet;
  - a comparison against Stata's `weakivtest`.
  - But 1.36 is far below any conventional threshold for the effective F.
- **For the main thread and Pedro: the paper's logit α (−0.064, p 0.092) is weakly identified by this measure.** This connects to the 10-16 weak-IV reset, whose battery was deleted and is to be rebuilt. IMPR can compute the MOP critical values (τ = 10 % and 20 %) and the weak-IV-robust interval for α if Pedro wants a formal statement.
- `scratchpad\netflow_1111\xcheck_effF.py` and `.json`.

**Decisions for Pedro:**
- **N6, BLP ext1 for the cap variants:** [IMPR: p4 and p3a only, if at all].
- **N7, the Findex fix:** [IMPR: fix at the next rebuild; correct %AE6, %AE7 and %AD7 now].
- **Weak-IV:** compute the critical values and a robust interval for the logit α? [IMPR: yes, it is local and light].

**K1:** the 300-market R-q pilot (headline, then savings-only, about 40–50 min) started at 14:53.

## Update 11-12 14:50: the two K2 plans you asked for (%AS6 and %AS2; your 00:28 point 2)

**The plan file:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\k2_plans_1112.md` (533 lines, md5 `315504f66a4d49959ca3cb82b2f408d0`).
- It was written read-only by a max-effort agent; each claim is tagged [V-code], [V-data] or [inf].
- Its probe is in `scratchpad\k2_plans_1112\`. It took one local probe of 9.7 min (0.44 GB).
- Nothing was edited, and nothing ran on Bouchet.
- **Line numbers** follow Pedro's 14:36 save of V_Friction: %AS2 is l.1388, %AS6 l.1396, the `tab:k2_breaks` note l.1438.

### Plan 1 (%AS6): run the random-coefficients break with the paper's own demand model (ext1)

**Today's step 7 is not even the extended stage's model as estimated. It has two defects; IMPR checked the first.**
1. **The wrong demographic draws.**
   - Step 7 (`diag_k2_nls_simulation.py` l.126) reads the local draws file of 07-27: `demo_draws_R2000_seed42.jls`, 13,957,120,024 bytes = 109,040 keys × 2000 × 8 demographics.
   - The I2 BLP used draws rebuilt on 11-05 (job 32207093): 111,650 keys, 14,291,200,024 bytes. IMPR confirmed both sizes.
   - Three demographics are standardized on different scales; for example, connections has mean 0.129 locally against 0.402 on I2 (the agent's figures).
2. **The D products.** Step 7 gives D products the local market's demographics. The engine gives each D product the national ("NATIONAL") draw, which is the same in every market.

**Also:**
- step 7 uses the extended stage's θ₂ (K > L, SE refused), not ext1's;
- it puts the random coefficients on top of the logit δ;
- it uses 200 of the 2000 draws.

**Which world is "the same demand model".**
- **(a), recommended: ext1's δ\* together with ext1's random coefficients,** integrated over the 2000 I2 draws.
  - With every product at δ\*, the integrated shares equal the data's, and so do the logit's.
  - So at κ = 0 the RC and logit inflows are identical by construction, and Panel A then measures only A1's error under the paper's own taste heterogeneity.
- **(b), the logit δ plus ext1's random coefficients,** is a hybrid.
  - The δ gap is 1.29 on average for type 4, and 0.9–3.3 for types 1, 2 and 5. It shifts the shares by type and market-quarter, so a move in κ̃ could not be attributed to heterogeneity.
- **The direction of the result under (a) cannot be predicted.** ext1's π(spread×gdp) = 0.946 per SD of gdp, so the price coefficient turns positive for consumers about 0.23 SD of gdp above the average market.

**Code** (each change needs Pedro's go; line tables are in the plan, §1.4):
- `diag_k2_nls_simulation.py`:
  - the constants at l.123–135: the result json becomes the ext1 json, the draws an I2 draws folder, RC_R 2000;
  - `rc_inputs` at l.887–923: read θ₂ from ext1 and check the stage;
  - `rc_simulate`: δ\*, and national keys for D;
  - `step7` and `main`.
- `diag_k2_netflow_simulation.py`: a new `read_delta_star`.
- `make_paper_numbers.py`: `K2_STEP7_DIR` and its description.
- **Six gates run before any fit,** above all:
  - the integrated shares reproduce the data to 1e-6;
  - at κ = 0 the RC and logit inflows are equal.

**Cost.**
- **Bouchet (recommended):** one task of about 1–1.5 h. The I2 draws, ext1's δ\* and the demand prep are already there.
  - The optional K2b fit of the change under random coefficients adds about 1 h. It would let the text drop "not simulated with random coefficients".
- **Locally:** only after a 14.3 GB download of the I2 draws, then 1.5–3 h.

**Keys:** 19 move.
- `kappa.k2.rc.{k05,k10}.rc.{kappa,se,bias,alpha,converged,at_box}`;
- `.inflow_ratio.{p10,p50,p90}`;
- `kappa.k2.rc.draws`, 200 → 2000.
- The truth, logit, `n` and `markets` keys stay.

**Text that depends on the result:**
- %D74 (l.836: "the level is lost", "upper bound", "neither converges");
- %AS1, %AS6 and %AS7, and Panel A of `tab:k2_breaks`;
- in the pending edits: %AS9, %AS10 and %D52, plus %AN40 if the K2b add-on runs.

### Plan 2 (%AS2): Monte Carlo standard errors for `tab:k2b_pairs`

**The noise (recommended):** independent normal shocks ξ on the type-4 mean utilities, SD 0.93.
- That is the within-cell SD of the logit type-4 utilities net of the gap, which is exactly the stored world minus the fe world.
- The probe's calibration: at this scale the simulated outcome varies within cells about as much as the real net flow (SD 0.00150 against 0.00143).
- **Rejected:**
  - resampling the simulated cells: the outcome is deterministic, so a resample is just the sandwich again, and 18 clusters are too few;
  - noise from K1's real-data residuals: that fit is unconverged.

**The design.**
- The table's four fits (fe and ideal worlds × before-after and dose), each warm-started from today's estimate.
- One ξ draw per replication, shared by the four fits.
- Seed `default_rng([20261112, r])`.
- **R = 100:** the SD is known to about ±7 %, and the mean to ±0.1 SD.

**Cost** (measured by the probe).
- The simulation takes about 1 s per replication. The noisy fits hit the inner cap: 3 outer iterations took 342 s.
- Estimate: 2–3.5 core-hours per replication. That is 15–30 core-hours for the pilot and 200–350 for the full run, about half a day as a Bouchet array. Not feasible locally.

**Code and keys.**
- A new `diagnostics/diag_k2b_mc.py`.
- A small refactor that pulls the K2b design out of `k2b_world`, which today writes a CSV.
- A collector `collect_k2b_mc`, and the table generator printing MC SDs under every pair.
- About 72 new keys, for example `kappa.k2b.sav.change_mcsd.fe_ba` and `_mcmean`, plus `kappa.k2b.mc.{R,xi_sd,…}`.
- A draft table note and a draft %AS2 paragraph are in the plan (§2.8).

**Main risk.** In the stored world, which has noise of about this size, no level fit converged. If i.i.d. ξ does the same, the result is a finding about the level rather than a standard error. Hence a pilot first.

### Shared setup, and a defect that affects the next K1 cluster run too
- **Two hard-coded paths will not resolve on Bouchet,** where results sit under `data/output/blp/` and `data/output/logit/`: `ALPHA_SOURCE` and `DELTA_BIN` in `diag_k2_netflow_simulation.py`.
- This also affects the next K1 cluster run, since the 11-09 and 11-11 changes.
- **The fix:** two environment overrides.
- `kappa/k1_job.sh` is the template for both plans' jobs.

### Decisions for Pedro (IMPR's recommendation in brackets; the agent's, endorsed by IMPR)
- **P1-a,** the world for the break [(a): ext1's δ\* with its random coefficients].
- **P1-b,** sorting of pool holders [none].
- **P1-c,** add the K2b fit under random coefficients [yes].
- **P1-d,** where Plan 1 runs [Bouchet].
- **P2-a,** the noise [ξ on type 4, normal, scale 1].
- **P2-b,** R [100].
- **P2-c,** a pilot first: 8 tasks at scales 1 and 0.3 [yes].
- **P2-d,** what the table prints [noiseless estimates, with MC SDs under every pair].
- **P2-e,** what to report if fits fail at scale 1 [Pedro's call after the pilot].
- **P2-f,** `tab:k2_breaks` keeps its sandwich SEs, with its own note sentence [keep].
- **P2-g,** numerical settings for the MC fits [the estimator's own].
- **The go for the code edits,** with bit-identity checks: the default runs must reproduce today's outputs.

**Overall** (agent and IMPR): Plan 1 under (a) with the K2b add-on, run beside the Plan 2 pilot after one shared setup. The full Monte Carlo only after Pedro's go on the pilot. Elapsed time is about 1 to 1.5 days.
- **Until then, the current `kappa.k2.rc.*` keys come from the July draws and the extended stage.** The text's description of the RC break should not claim it uses the paper's demand estimates.

## Update 11-12 14:15: K1 R versions, with t−1 timing and fill B done; both 50-market smokes run, and both are degenerate

**The code.** `kappa/k1_netflow_nls.py` `c32b894d` → **`8920a40a6e145f60fee2299b9a012802`** (CR 0). Backups are in `scratchpad\k1_lambda_1111\`.
- The change is confined to `registry_lambda`, its docstrings and the stop message.
- **Timing:** λ(j, t) is now the version's series at (j, t − 1), membership included, in all four R versions (Pedro, 00:28).
- **Fill B,** where the file has j but not t − 1:
  - membership follows the registry's own rule: carried from j's last quarter, or held back from its first;
  - the value is the version's segment value at t − 1 from `registry_segment_quarter.csv`;
  - R-fixmem uses its fixed segment.
  - `registry_segment_quarter.csv` joins the md5-manifest guard.
- **What is still missing at t − 1: nothing.**
  - Kirton 2016Q4, Master 2018Q2 and Neon 2018Q2 now resolve directly from the file.
  - **One fill remains:** the 64 keys (j, 2015Q4) behind the 2016Q1 rows (63 B, plus 1 D at λ = 1). They precede the file, so nothing can be carried and the held-back branch fills them. Under every version that equals the 2016Q1 value, so nothing changes against the unlagged series. **Pedro to confirm.**
- **The B-row mean λ on K1 rows,** t → t − 1:
  - R-q 0.8598 → 0.8597;
  - R-const 0.8589 → 0.8589;
  - R-fixmem 0.8585 → 0.8586;
  - R-annual 0.8747 → 0.8758.
- **No registry series and no `reg.lambda.*` key changes,** because the lag lives in K1. The two CSVs K1 reads are byte-identical (58696c86 and be25a38b).
- **Tests:**
  - all four versions equal an independent `merge_asof` reference;
  - the guards refuse another panel and a manifest mismatch;
  - **S-stmt is still bitwise equal to `99eeee52`** (`compare_sstmt.py`: inputs, the frame, the three pilot forms, the 50-market draw and the objective evaluations).
  - Test 1, which passed on `c32b894d` last night, was not rerun. Today's edit is in code test 1 does not execute, and the S-stmt comparison is bitwise.

**The 50-market R-q smokes.** Both use before_after, `--exit-by-cell --trim p99`, starts 0, 0.5, 1 and 1.5, maxit 500, and 9,999 draws. n = 2,570 rows, G = 9 (G\* 5.37), 21 markets. Nothing was written to Rout.

| form | runtime | best SSR/yy | stop | at the box | α (se) | the rest |
|---|---|---|---|---|---|---|
| headline | 115 s | 0.1266 (start 1) | maxit 500, not converged | **κ_td 10 (upper)**, κ_chk −4 (lower) | −0.247 (0.051) | κ −4.0 (se 2392), dκ −9.7 (4699), ln E_c −14.0 (2392) |
| savings-only | 261 s | 0.1237 (start 0) | maxit 500, not converged | **κ_td 10 (upper)** | −0.117 (0.031) | κ 1.81 (52.9), dκ −7.34 (70.4), ln E_c −8.19 (53.0) |

- **These are not usable estimates.** In every start of both forms, some parameter ends at its box.
- **The time-deposit pair runs to its upper box.** e^−10 means the rivals' daily pool gets almost no weight. κ, dκ and ln E_c have huge standard errors.
- **The R-q path works end to end** and now estimates κ_td, dκ_td and ln E_c. With 21 markets, though, it is not identified.
- **Process note.** A frame-building step of the net-flow agent overlapped the last ~2 minutes of the headline smoke. Both agents had matched `python` and missed the venv's `python3.11` worker; both were corrected. This affects that smoke's runtime only.

**Pedro (11-12, about 14:40): both suggestions accepted.**
- The 2015Q4 held-back fill stands.
- The 300-market R-q pilot (headline and savings-only, the S-stmt pilots' design, outputs to scratch) runs after the cap-sensitivity logits.

**The decisions, as they were put to Pedro:**
1. **The held-back fill for 2015Q4** (equal to the 2016Q1 value) [IMPR: accept].
2. **The box-bound time-deposit pair** before any larger R run. IMPR recommends the 300-market R-q pilot in the same design as the S-stmt pilots (headline and savings-only, 4 starts), so the comparison is like for like.
   - It is local, about 1–2 h. It runs after the cap-sensitivity logits now running.
   - If κ_td still hits its box at 300 markets, the pair is not identified in K1's design. The R versions would then report κ_td as a bound, as the paper does for the pilots.

## Update 11-12 14:10: net-flow Part A done; your outside-share reconciliation; Stage 1 logits running

**Part A** (`diagnostics/diag_netflow_gap.py` `a790c191`, a new script; outputs in `Rout/diag_netflow_gap/`, md5s in its `MD5SUMS.txt`; run 13:38–13:41).
- **Gates.**
  - **G-A1:** the rebuilt K2 world reproduces `k2_numbers.json` with max abs diff 0.0, on the closed-form δ the a216 run used (the binary δ gives 3.9e-14).
  - The ω-aware copy of `simulate` equals K2's bit for bit.
  - K0's net flow, rebuilt in-process, equals `k0_netflow_panel.parquet` exactly.

**A.1: |net flow| / carried stock, type 4** (306,349 of the key's 310,839 cells have a unit in the world):

| | p25 | p50 | p75 | p90 | stock-weighted |
|---|---|---|---|---|---|
| **data** (`kappa.k0.moving.median.k4`) | 0.042 | **0.102** | 0.215 | 0.425 | 0.091 |
| W0, the K2 world | 0.0001 | **0.010** | 0.040 | 0.158 | 0.055 |
| W0 at κ = 0 | 0.0001 | 0.010 | 0.041 | 0.161 | 0.056 |
| W1, stored δ at κ = 0 | 0.0001 | 0.010 | 0.042 | 0.153 | 0.054 |
| W2, awake money from the capped φ | 0.010 | 0.027 | 0.126 | 0.350 | 0.066 |
| W3, stored δ at κ = 0, capped φ | 0.010 | 0.028 | 0.127 | 0.328 | 0.064 |
| W4, the statements' ω | 0.0000 | 0.0033 | 0.041 | 0.174 | 0.050 |

**By cell group** (type-4 medians):

| cells | share | data | W0 | W1 | W3 |
|---|---|---|---|---|---|
| on the menu, φ̂ ≤ 0.99 | 27 % | 0.076 | 0.072 | 0.076 | 0.076 |
| on the menu, φ̂ > 0.99 | 32 % | 0.128 | 0.0010 | 0.0011 | **0.120** |
| off the menu (truncated by 14-B) | 41 % | 0.101 | 0.0035 | 0.0035 | 0.0100 |

- **κ does not cause the gap.** W0, W0 at κ = 0 and W1 are almost identical.
- **The world matches the data where its assumptions hold.** That is the on-menu cells with φ̂ ≤ 0.99: 0.072–0.076 against the data's 0.076.
- **The gap comes from two cell groups.**
  - **Capped cells (32 %).** The world takes awake money from the raw φ̂, which removes the flow. With the capped φ that the shares use (W3), the world gives 0.120 against the data's 0.128.
  - **Truncated cells (41 %).** In the data the outflow there exceeds the awake stock, which the model cannot produce; the world is bounded at (1 − φ)·carry ≤ 0.01.
- **So the paper's "2.1 % awake, 14 % stays, against a 10 % median net flow" gap is mostly built into the simulated world, not behaviour.** It comes from the raw-vs-capped φ inconsistency and from the cells (14-B) truncates. Awake money is 2.10 % with the raw φ̂ and 2.28 % with the capped φ.
- **Type 2:** the data medians are 0.033; the world gives W0 0.005 and W3 0.016.

**A.2: the share of Var(NF/carry) that lies within the market-quarter, across banks:**

| type | carry-weighted | unweighted, winsorised | carry-weighted, winsorised |
|---|---|---|---|
| 4 | **0.855** | 0.573 | 0.694 |
| 2 | 0.824 | 0.463 | 0.404 |
| 1 | 0.980 | 0.608 | 0.610 |

- Net flows are mostly reallocation across banks, not common market movements.
- For type 4, carry-weighted, the national-quarter part is 0.016 and the market part 0.129.
- With two or more banks in the market-quarter, the within share is 0.885. 38,541 of 112,127 type-4 market-quarters have a single bank.

**A.3: are large net flows rebooking events?** No. The top decile (|NF|/carry > 0.425) is compared with all cells; lift is the ratio of the two.

| event | top decile | all cells | lift |
|---|---|---|---|
| branch change, as reported | 0.005 | 0.015 | 0.35 |
| branch change, static map | 0.016 | 0.014 | 1.18 |
| CNPJ set change in the bank's own deposit line | 0.006 | 0.001 | 9.5 (rare) |
| change in the conglomerate's members | 0.169 | 0.206 | 0.82 |
| BB and Caixa judicial-deposit cells | 0.548 | 0.474 | 1.15 |
| any event | 0.676 | 0.663 | 1.02 |

**A.4: feasibility** (pooled over types 1, 2 and 4):
- **The outflow side** at ω = 1 is the (14-B) truncation, which holds in **0.402** of cells (0.337 carry-weighted).
  - It equals K0's truncation flag on every cell.
  - It reconciles to `demand.flows.trunc_share.E3` 0.4546 as follows: types 1, 2 and 4 only give 0.431; K0's cells with φ̂ and a carry give 0.422; K0's mask gives 0.402.
  - With the statements' ω, type 4 goes from 0.413 to 0.453.
- **The inflow side** is zero by construction with the capped φ.
  - Against the raw φ̂, 0.159 of cells exceed it, but only 0.041 carry-weighted.
  - Against the market size before the floor: 0.0017 of cells, 0.043 carry-weighted.
- **Two choices the agent made, for Pedro to confirm:**
  - ω timing in W4 follows K1: the statement of year y applies to the quarters of y + 1;
  - floor markets are the 4,068 where M is above the anchor. The share-sum test finds 4,043, because 25 floor markets lost a row in `finalize`.

**Your reconciliation: `kappa.k0.outside.share_Mhat.median` 0.971 against `menu_share.median` 0.863.**
- **Both are medians over the same 111,146 market-quarters,** so the kind of median contributes nothing.
- **The chain from one to the other:**

| step | median | change |
|---|---|---|
| the key: 1 − (B lagged stocks / M̂) | 0.971 | |
| the numerator becomes the active balance | 0.999 | +0.028 |
| the denominator becomes (1 − min(φ̂, 0.99))·M̂: **the capped φ** | 0.921 | **−0.078** |
| D products enter the menu | 0.864 | **−0.057** |
| K2's type-4 δ (= `menu_share`) | 0.8635 | −0.0002 |

- **The capped φ is the largest step.** At the median, 1/(1 − φ_c) = 100.
- **D balances in stock terms** account for only −0.008: netting D firms' balances, allocated across markets by M̂, gives 0.962.
- **The deposit-weighted mean of O/M̂ is 0.732;** the M̂-weighted mean is 0.911.
- **So 0.971 is right as defined.** The gap is definitional: one is a stock share of the total market, the other a logit share of the active market. It is mostly the capped φ (−0.078) and the D products (−0.057). No key changed.
- Files: `scratchpad\netflow_1111\recon_outside.py` and `.json`.

**Part B, Stage 1.**
- **Done:**
  - B-S0, the Julia environment;
  - B-S1, 24 frames (12 variants × E3/E4, 1.3 GB in `C:\Users\pedro\AppData\Local\Temp\capsens_1111\`);
  - G_t > 0 in every frame (minimum 0.144, E4 p3b). The frames match the dry forecast exactly.
- **The logits (B-S2) started at 14:07,** about 1.5–2 h, as the only heavy job.
- **Variant f is reported next to production and base\*.** Its pre-logit numbers: E3 median outside share 0.859, against 0.864 in production and base\*; mean ρ(1 − s) 3.814 against 3.818. Production α is E3 −0.0641 (0.0315) and E4 −0.0637 (0.0321).
- The driver is `diagnostics/diag_cap_sensitivity.py` (`bfd2124d`, a new script).
- **A flag, NOT YET VERIFIED.** The logit's type-4 first stage gives a clustered Wald F of 22.3 (E3) and 25.0 (E4), but a Montiel Olea–Pflueger effective F of **1.36 / 1.34** (type 5 about 2.2).
  - The effective F is the agent's own implementation, checked only through the Frisch–Waugh–Lovell identity.
  - IMPR asked for an independent cross-check before it counts as a finding.
  - If it holds, the logit α (and the weak-IV caveat) needs the main thread's and Pedro's attention.

## Update 11-12 13:40: IF.data register re-fetched (segment map holds); equation macros regenerated; Bouchet not yet launched

**1. The IF.data register, re-fetched read-only (your 18:25 item; the service answered again at 13:25).** All 36 quarters, 2016-03 to 2024-12, went into scratch (`scratchpad\ifdata_refetch_1112\dl_20261112\`).
- **The BCB has revised the register since the pipeline's 2026-04-07 download.** No quarter is byte-identical.
  - `Sr` differs on 9,242 institution-quarters: 5,540 gained a segment (for example most of 2017-03, which was empty before), 3,500 lost one, and 202 changed.
  - Institutions: 117 appear only in the old copy, 75 only in the new.
- **What it does to our segment map** (`make_lambda_registry.py` rerun in scratch on the new register; production untouched; `scratchpad\ifdata_refetch_1112\map_on_refetch.out`):
  - **11 of 2,277 B conglomerate-quarters change segment, holding 0.054 % of B time deposits:**
    - Bank of China S3 → S4 (2022Q1–2023Q4);
    - Banpará S4 → S3 (2019Q1–Q4, two quarters in the panel);
    - Modal S3 → S4 (2022Q2);
    - Banco Pleno S3 → S4 (2023Q3).
  - No S1 or S2 bank changes. 62 B rows that were "held back" now take their own row's segment, with the same segment.
  - **The keys:** S1 0.868 / 0.924 and all.min 0.106 come from the registry, so they are unchanged. `reg.lambda.B.tdw.2019Q4` moves 0.7988 → 0.7990 and still prints 0.80.
  - **The B-weighted λ series** move by at most 0.0015 in any year and version. The launch-window change stays +0.018 (R-q) and −0.006 (R-const).
- **IMPR's recommendation:** keep the archived copy, so the map stays byte-consistent with the pipeline's register and the panel's own `segment` column (7,437 of 7,437 agree).
  - Archive the re-fetch in DATA_RAW as a dated second vintage, with a source-log row and a NOTES line that says it changes 0.05 % of B time deposits.
  - Pedro's call.

**2. `make_paper_equations.py` rerun on Pedro's 13:32 compile** (which carries the new attention term).
- Before: 31 equations and 116 labels (11-09). Now: 42 equations and 151 labels.
- New: eq:22 (openings) and the tax-clock equations eq:d1–d6.
- eq:15 and eq:20 now print `\left`/`\right`, as in the paper.

**3. Bouchet: D0 submitted at about 13:40 (Pedro).**
- **The checks Pedro pasted:**
  - the cluster's base was the files the edit started from: `blp_ext1_starts.sh` 75ad5510, `blp_engine_cpu.jl` 2b467eb6, `blp_engine_gpu.jl` 16261cde;
  - the five `scripts/` uploads and the two staged engine files match their md5s;
  - CR 0 in both `.sh` files.
- **Jobs:** `d0_ext1_E3` **33145172** and `d0_ext1_E4` **33145175**, on the production engine, read-only.
- **R0 submitted about 13:50: job 33145739.**
  - The build printed "E4_x1zero: delta start = blp_checkpoint_E4_spec_12_ext1.bin (f7ecf3b16b47566e...)" and "sandbox OK".
  - The staged engine was copied in (7d34eaa7 / c5b9c692 verified in the sandbox).
  - Start x1zero (0, 1.0272, 0.0914, −0.026, −0.1231, 0), at default flags.
- **The 12 validation fits wait** until the D0 summaries are read (r0).
- **The reads** are the D0 summary block per log, R0's `Q_value`/`theta2` against `diag_ext2_20261106`, and R0's `--report`.

## Update 11-12 13:30: your 23:54 and 00:28 read; work started

All of it is local, and only one heavy job runs at a time:
1. **K1, Pedro's two decisions** (resumed agent):
   - fill the three registry gaps with option B, the carried segment;
   - read the registry share at t−1 in all four R versions. The lag sits in K1, so the registry series and the `reg.lambda.*` keys should not change; IMPR will confirm.
   - Then rerun the bitwise checks, and run the 50-market R-q smoke in the headline and savings-only forms.
2. **Net-flow Part A** (resumed agent): check the script against the plan, pass gate G-A1, then the full run (about 20 min).
   - **During that run:** your reconciliation of `kappa.k0.outside.share_Mhat.median` (0.971) with `menu_share.median` (0.863). It decomposes the gap into D balances, the capped φ in M̂, and the median over market-quarters against the median market. It also gives the deposit-weighted mean of O/M̂. No key changes.
   - **Then:** Stage 1, with variant f next to base.
3. **The two plans** for %AS6 (the RC break with ext1's coefficients and δ\*) and %AS2 (Monte Carlo SEs for the K2b fits). A new agent writes them read-only to `scratchpad\k2_plans_1112.md`. No run without Pedro's go.
4. **The IF.data register is back** (HTTP 200 at 13:25). IMPR is re-fetching the 36 quarters read-only into scratch and comparing `Sr` with the archived copy that the segment map uses.
5. **Not started, waiting for Pedro's go:** the `tab_tax_clock` Panel B header, which is cosmetic.

## Update 11-11 23:41: K1 test 1 passes on the `--lambda` script

- **The run.** `kappa/k1_netflow_nls.py` `c32b894d` (the `--lambda` edit, default S-stmt), test 1 with `--recovery Rout/k2_fa38794d_a216/k2_nls_recovery.csv` on the a216 context, 23:26–23:40 (851 s).
- **"passed":**
  - max |estimate − stored| **5.55e-17**;
  - standard errors against stored **3.9e-15**;
  - the bootstrap's (Jt, r) reproduce V to **1.6e-15**.
- **Against the 11-09 run on the old script `99eeee52`** (`kappa_1109\test1_a216\k1_numbers.json`): **all 19 fields of the test-1 block are bitwise equal**, including:
  - α −0.21590468572391625, κ 0.49808232812697567, κ_chk 0.2934799301504663;
  - κ_td 0.4605112113988011, a 0.46978712525281824, a_null −0.290770136034226.
- **So the edit leaves the estimator unchanged at the default.** Together with the agent's checks (bitwise-equal inputs, and a pilot refit bitwise equal), the S-stmt path is today's.
- **Outputs:** `scratchpad\k1_lambda_1111\test1\`; nothing in Rout changed. The comparison script is `scratchpad\k1_lambda_1111\compare_test1.py`.
- **One small flag.** Test mode writes no top-level `lambda_version` in its `k1_numbers.json`; the fit path does (checked by the agent on 15-market runs). Not a problem for test 1, which always uses S-stmt.
- **Still open:** the R-q smoke waits for Pedro's choice of fill for the three registry gaps (update 23:40, point 4; IMPR recommends B, the carried segment).

## Update 11-11 23:40: the advertising test done; `all.min` changed; bib entry added; Bouchet lines (D0 + R0); state at shutdown

**Context.** VS Code was closed by accident at about 23:21, which stopped the three agents. IMPR checked what each left and finished what could be finished. Pedro shuts the PC down at about 00:05.

**1. The advertising difference test: done.**
- **The script:** new `diagnostics/diag_p4_adv_group_diff.py` (`a9bdf589`).
  - All its gates pass, including an exact replay of P4's bootstrap (p 0.598, interval [−0.2282, 0.1536]) and the CR2 unit tests.
  - It writes `Rout/p4_adv_group_diff.json`.
- **The collector:** `collect_pix_adv` in `make_paper_numbers.py` was edited by the agent (`4af89d6f` → `da91d0f7`).
- **The rebuild:** 416 → 419 entries; the 3 new keys are ok; **no existing value changed** (checked by IMPR against the agent's snapshot).
- **Results** (rest minus the five largest; RI = randomization inference over the 42,504 five-sets of the 24 steady advertisers):

| contrast | estimate | CR1 se | wild bootstrap p | **RI p (t)** | RI 95 % set | CR2 p (df) | G_eff; Caixa's share |
|---|---|---|---|---|---|---|---|
| **pooled** (`pix.adv.diff.*`) | **0.019** | 0.037 | 0.598 | **0.849** | [−0.150, 0.206] | 0.676 (1.60) | 1.80; 0.737 |
| pooled, without Caixa | 0.031 | 0.118 | 0.795 | 0.918 (8,855 four-sets) | — | 0.811 (3.23) | |
| pooled, big four (Caixa in the rest) | 0.025 | 0.116 | 0.828 | 0.931 (10,626 four-sets) | — | 0.846 (3.23) | |
| **intensive margin** (`pix.adv.diff_int.*`) | **0.190** | 0.070 | 0.116 | **0.061** | [−0.007, 0.377] | 0.134 (1.72) | 1.92; 0.714 |
| intensive, without Caixa | 0.224 | 0.167 | 0.192 | 0.518 | — | 0.258 (3.43) | |
| intensive, big four | 0.207 | 0.163 | 0.217 | 0.523 | — | 0.281 (3.41) | |

- **The five largest's own elasticity:** the CR2 p (`pix.adv.big5.elast_p_cr2`) is **0.132**, against the bootstrap's 0.097.
- **For A5, the Caixa sentence.** Without Caixa the point estimate barely moves (0.031 against 0.019), but its standard error triples (0.118 against 0.037).
  - So **the precision rests on Caixa, not the estimate.** Caixa holds 74 % of the identifying variation.
  - "Rests mostly on one conglomerate" is right about the precision, not about the sign or size.
- **The keys you asked for are all ok:**
  - `pix.adv.diff.elast` 0.019, `.p_ri` 0.849, `.ri.lo` −0.150, `.ri.hi` 0.206;
  - `.G.big5` 5, `.G.rest` 63, `.caixa_share` 0.737;
  - `pix.adv.big5.elast_p_cr2` 0.132.
  - The macros carry them.

**2. `reg.lambda.pfpj.all.min` now leaves out S5's phase-in: 0.065 → 0.106 (S4, 2023-12); `\pn[2]` prints 0.11.**
- `make_lambda_registry.py` `cd2d60d7` → **`0717383b`**: the key leaves out months before FIRST_FULL, and the json records `first_full_month`.
- `collect_lambda_registry` in `make_paper_numbers.py` applies the same rule (34 Decembers): `da91d0f7` → **`457a9439`** (CR 0). Backups are in `scratchpad\edits_1111\`.
- **The rebuild (23:36):** rc 0, **0 MISMATCH**, 419 entries (ok 340). **Only `reg.lambda.pfpj` changed.** S1 stays 0.868 / 0.924, and the B time-deposit-weighted λ stays 0.799.

**3. References.bib: `mackinnon2020randomization` added** (Pedro's go; Google Scholar layout, next to `mackinnon2017wild` and `mackinnon2023cluster`).
- **The entry:** MacKinnon, James G. and Webb, Matthew D. (2020), "Randomization inference for difference-in-differences with few treated clusters," *Journal of Econometrics* 218 (2), 435–450, doi 10.1016/j.jeconom.2020.04.024. The fields were checked on EconPapers / RePEc.
- **The file:** 906 → 907 entries; CRLF kept; braces balanced.
- **biber** (tool mode, whole file): 0 warnings, 0 errors, before and after.
- **To do:** the citation is the main thread's, `\textcite{mackinnon2020randomization}`.

**4. K1 `--lambda`: the edit is finished and verified; test 1 is running.**
- **The script:** `kappa/k1_netflow_nls.py` `99eeee52` → **`c32b894d`** (CR 0). It compiles.
- **The default (S-stmt) is bitwise today's.** The agent checked the ω/λ arrays, the 313,165-row frame and every model array for the three pilot forms and the 50-market draw.
- **A stored pilot refits bitwise:** savings-only start 1 gives SSR/yy 0.2197953547531662, dκ̃ −0.3410396990019272, all six standard errors, 48 iterations, and stop "no descent step".
- **The R versions:** λ is read by (conglomerate, quarter) with two guards (the panel md5 and the manifest). They estimate κ_td, dκ_td and ln E_c, and write `lambda_version`.
- **Test 1** started at 23:26 (about 23 min). Its result is posted below, or tomorrow.
- **The 50-market R-q smoke is blocked by a gap in the registry, and Pedro decides.**
  - Three conglomerate-quarters hold a lagged stock but are absent from the registry:
    - Kirton Bank 2016Q4: R$ 37.0 bn, reaching 23 rows in 6 markets of the 50-market draw and 126 rows of the 300-market draw;
    - Master 2018Q2: R$ 101 m;
    - Neon 2018Q2: R$ 146 m.
  - **The options:**
    - **(A)** the conglomerate's own last registry value;
    - **(B), recommended:** carry its segment, the registry's own "carried" rule, and take that segment's value at t;
    - **(C)** stop, as now;
    - **(D)** drop the rows they reach.
  - For Kirton, A and B agree: R-q 0.883.
- **A timing flag.** R-q reads the share at the end of quarter t, while the pools are the stocks of t−1. Reading at t−1 has no look-ahead, matches the statements' timing, and would also remove the missing keys. Main thread and Pedro.

**5. Net-flow Part A: not run tonight.**
- `diagnostics/diag_netflow_gap.py` (`a790c191`) compiles and has A.1–A.4 and `main`. The agent ran smokes of A.1–A.4 on subsets (`scratchpad\netflow_1111\smoke_*`).
- Neither the full run nor gate G-A1 ran. The Stage 1 driver (`diag_cap_sensitivity.py`) does not exist yet.
- **Tomorrow:** check the script against the plan, run gate G-A1, then the full run (about 20 min). Then Stage 1, with variant f next to base.

**6. Bouchet: ready for D0 (in full) and R0, per Pedro's go on IMPR's recommendations.**
- **The 12 validation fits wait** until D0 is read, because D0's scan sets the step limit r0.
- **D3** (adopt only on the FD check) matters only at adoption. It gets coded before any adoption.
- **Promotion** of the engine into `scripts/` comes after R0 reproduces 11-06. The flags stay off until the validation passes.
- **Upload** (web file manager, from the local repo root; md5s re-checked 23:06):

| file | md5 | destination |
|---|---|---|
| `blp_diag_ext1_d0.jl` | `02c8d7237e226156c6ee0c28d91773f7` | `~/project_pi_mf2263/pf382/dep_comp/scripts/` |
| `blp_diag_ext1_d0_job.sh` | `87ea35a9df7fb9bcee7b092809cac862` | same |
| `blp_ext1_starts.sh` (replaces `75ad5510`; `--report` only) | `f6920e09dcdfa7d438b23085c63c96f5` | same |
| `blp_ext1_starts_val1112.csv` | `3f638a0635bb1f816a8af987f1f7cd44` | same |
| `blp_ext2_starts_r0def1112.csv` | `4881d7601fe0ea033ac63deb690be88c` | same |
| `blp_engine_cpu.jl` | `7d34eaa7a8b000826f687925ef4e752f` | `~/scratch_pi_mf2263/pf382/engine_b_1111/` (staging, **not** `scripts/`) |
| `blp_engine_gpu.jl` | `c5b9c692cad0110a0e824d02de6b9a91` | same staging folder |

- **The lines** (each one line; cluster-only, none tested on Bouchet yet):
  - **C0, the staging folder, before the upload:** `mkdir -p ~/scratch_pi_mf2263/pf382/engine_b_1111`
  - **C1, check `scripts/`.** It must print the five md5s above, then `0` and `0`:
    `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && md5sum blp_diag_ext1_d0.jl blp_diag_ext1_d0_job.sh blp_ext1_starts.sh blp_ext1_starts_val1112.csv blp_ext2_starts_r0def1112.csv && tr -cd '\r' < blp_diag_ext1_d0_job.sh | wc -c && tr -cd '\r' < blp_ext1_starts.sh | wc -c && chmod +x blp_diag_ext1_d0_job.sh blp_ext1_starts.sh`
  - **C2, check the staging folder.** It must print `7d34eaa7…` and `c5b9c692…`:
    `cd ~/scratch_pi_mf2263/pf382/engine_b_1111 && md5sum blp_engine_cpu.jl blp_engine_gpu.jl`
  - **D0-E3** (prints a job id; cancel with `scancel <id>`):
    `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -J d0_ext1_E3 -o logs/%x_%j.out -e logs/%x_%j.err --export=ALL,DIAG_ROUTINE=3 blp_diag_ext1_d0_job.sh`
  - **D0-E4:**
    `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -J d0_ext1_E4 -o logs/%x_%j.out -e logs/%x_%j.err --export=ALL,DIAG_ROUTINE=4 blp_diag_ext1_d0_job.sh`
  - **R0-build** (it must end with "E4_x1zero: delta start = blp_checkpoint_E4_spec_12_ext1.bin (f7ecf3b16b47566e...)" and "sandbox OK"):
    `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && EXT_STAGE=ext2 EXT_TAG=r0def1112 EXT_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_starts.sh --build-only`
  - **R0-engine** (it must print `7d34eaa7…` and `c5b9c692…`):
    `cd $HOME/scratch_pi_mf2263/pf382/diag_ext2_r0def1112 && cp -p ../engine_b_1111/blp_engine_cpu.jl ../engine_b_1111/blp_engine_gpu.jl E4_x1zero/scripts/ && md5sum E4_x1zero/scripts/blp_engine_cpu.jl E4_x1zero/scripts/blp_engine_gpu.jl`
  - **R0-submit** (one job id; default flags, no `BLP_*` set):
    `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && EXT_STAGE=ext2 EXT_TAG=r0def1112 EXT_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_starts.sh --submit-built`
  - **R0-cancel:** `cat $HOME/scratch_pi_mf2263/pf382/diag_ext2_r0def1112/E4_x1zero/submitted.jid | xargs -r scancel`
  - **Status:** `squeue -u $USER -o "%.10i %.22j %.8T %.10M %R"`
- **Tomorrow, the reads:**
  - **D0:** paste the output of `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sed -n '/=== D0 SUMMARY/,/=== END D0/p' logs/d0_ext1_E3_*.out` and the same for E4.
  - **R0:** paste the output of `grep -o '"Q_value":[^,]*\|"theta2":\[[^]]*\]' $HOME/scratch_pi_mf2263/pf382/diag_ext2_20261106/E4_x1zero/data/output/blp/blp_results_E4_spec_12_ext2.json $HOME/scratch_pi_mf2263/pf382/diag_ext2_r0def1112/E4_x1zero/data/output/blp/blp_results_E4_spec_12_ext2.json`. Q and θ₂ must agree, to the bit or to about 1e-12 if the GPU reductions are not reproducible.
- **Cost:** D0 about 4–6 GPU-h; R0 about 1–2.5 GPU-h; both finish without the PC.

## Update 11-11 23:14: your 22:55 and 23:00 read; cleanup verified; the BLP engine code is done (local)

**1. The scratch cleanup is verified; item 10 is closed.**
- **How it was checked.** The script is gone, deleted at Pedro's request. So the check used its 20:07 dry-run listing, `cleanup_dryrun_1111.txt`, which has the same 123 targets. The check script and its output are `scratchpad\cleanup_check_1111\verify_cleanup.py` and `.out`.
- **The result:**
  - all 123 targets are gone, and their 14 parent folders remain;
  - all 43 kept items checked are present.
- **The kept items checked:**
  - `rebuild_1104`;
  - `kappa_1105`, `kappa_1108` and `kappa_1109`, with both context pickles;
  - `refresh_i2_1106` (`key_table.md`, `final_report.md`), `gfb_1108\before_R3`, `ssel_1108`, `e4fix_1106\new`, `tools`, and the loose I2 files;
  - every 11-11 working folder and plan; 904ee55d `panel_52ea6495`; `tasks\`;
  - in the older root: `i6`–`i9bb`, `ibge_scripts`, the exact-IFT drafts, and BB's `bd1` and `b7`.
  - The BB session keeps its 13 scripts and the K2 session its 32. The Pix session is untouched (97 + 93 MiB).
- **Sizes now:** the IMPR scratchpad 883 MiB, the K2 session 190 MiB, the older root 421 MiB; C: has 84.1 GB free.
- **The check verifies itself:** it uses the `\\?\` long-path prefix, and it first requires that prefix to resolve a path that exists and reject one that does not.

**2. `reg.lambda.pfpj.all.min` without S5's phase-in (your decision): the edit waits for one agent.**
- **It takes two edits:**
  - the key in `make_lambda_registry.py`;
  - the recomputation in `make_paper_numbers.py`. `collect_lambda_registry` recomputes the December shares and requires 35 rows that match the key; it will expect 34, with S5 counted from 2019-12.
- `make_lambda_registry.py` will also record the first full-stock month (S5: 2019-03) in its json. The collector will read the rule from there instead of repeating it.
- **Why it waits.** The advertising agent is editing `make_paper_numbers.py` now, and its rebuild requires every existing key to stay unchanged. So IMPR makes both edits right after that agent reports, then reruns both scripts.
- IMPR will post when the macro changes: `\pn[2]{reg.lambda.pfpj.all.min}` 0.07 → **0.11** (S4, 2023-12).

**3. Your text requests went to the agents at 23:10:**
- **The net-flow agent:** variant f goes next to base\* and production in the Stage 1 table, with α, se, the mean elasticity and the median outside share, for E3 and E4.
- **The advertising agent:**
  - the eight keys the sentence quotes, all built at status ok;
  - the rows without Caixa and for the big four, each with se and p.
- **The MacKinnon and Webb (2020) entry for References.bib:** waiting for Pedro's go.

**4. Item 7, the BLP engine: the local work is done.**
- The agent reported at 23:03; its log is `scratchpad\blp_engine_impl_1111\log.md`. Nothing was uploaded or submitted.

- **The flags.** Each is an environment variable read at call time, like `BLP_FD_CHECK`.
  - **b1, `BLP_OUTER_STOP_REASON`, on by default.**
    - `solve_box_lbfgsb` runs L-BFGS-B's own loop, with the same Fortran calls in the same order.
    - Per start, it records stop_reason, line_search, n_iter, n_evals, f_start, f_at_x and moved.
    - `=0` restores today's call.
    - The first four return values are unchanged; `converged` still means fewer than 500 evaluations.
  - **b2, `BLP_OUTER_STEP`** off/lite/full (default off; `BLP_OUTER_R0` 0.02). "lite" limits the first trial to r0 per coordinate; "full" adds a moving box.
  - **b3, `BLP_INNER_GUARD`** (default 0). Each contraction starts from the last converged δ. A failed evaluation returns a finite penalty Q and restores δ.
  - **b4, `BLP_INNER_CLAMP_TEST`** (default 0). A contraction whose δ sits at the ±500 clamp, with a residual left, counts as not converged.

- **Tests, CPU, local.** The toy test passes 31/31. On the real E3 objective (two quarters, R = 50):
  - **A:** the defaults reproduce today bit for bit.
  - **B:** b3 is bit-identical when nothing fails.
  - **C:** b4 flags a clamped δ that today calls converged.
  - **D: from a start that freezes today, lite reaches Q 0.0486 from 0.627.** Lite + b3 follows the same path, and full + b3 reaches 0.154.
  - **Not testable locally:** the GPU paths, D0 itself (parse and unit checks only), and the sandbox flow.

- **Two findings for the record:**
  - **Freezes do not end "ABNORMAL"; this corrects the plan's inference.**
    - The line search shrinks its step to zero.
    - L-BFGS-B then re-evaluates θ₀ from the polluted δ, accepts it, and stops on an ordinary convergence message.
    - So no gate can rest on the stop reason alone. A freeze shows as moved = 0 with f_at_x above f_start, and in the FD check.
  - **b3 throws away a failed contraction's partial progress.** So a far start can still stop at θ₀ ("first_eval_failed"). That is the plan's "unreachable" case, which needs the continuation δ.

- **Files** (LF md5, CR 0; IMPR re-checked all eleven):
  - `blp_engine_cpu.jl` `2b467eb6` → `7d34eaa7`;
  - `blp_engine_gpu.jl` `16261cde` → `c5b9c692`;
  - `blp_ext1_starts.sh` `75ad5510` → `f6920e09`. Only `--report` changed: it now prints FD, moved, δ start, stop, evals and guard columns for every rung;
  - new: `blp_diag_ext1_d0.jl` `02c8d723`, `blp_diag_ext1_d0_job.sh` `87ea35a9`, `blp_ext1_starts_val1112.csv` `3f638a06`, `blp_ext2_starts_r0def1112.csv` `4881d760`;
  - local only: `diagnostics/test_solve_box.jl` `c2cb9dbe`, `diagnostics/smoke_engine_flags.jl` `8dd5a4d0`.
  - **Not touched:** `converged`, TOL_OUTER, the gates, the CPU-IFT engine.

- **Prepared for the next Bouchet session, none run.** The single-line commands, with their checks and cancel lines, are in the log.
  - **D0:** two jobs, about 4–6 GPU-h in full. `DIAG_PARTS=i+iii+iv` skips the scan (ii) and brings it to about 3.
  - **The validation of (b):** 12 fits (E3/E4 × control, warm_rc4, cold, rand1–3) with lite + b3 + b4, about 12–30 GPU-h. The edited engine is staged outside `scripts/`, so production keeps today's engine.
  - **R0, optional:** the edited engine at default flags re-runs the 11-06 E4 ext2 x1zero fit and must give the same Q and θ₂. About 1–2.5 GPU-h.

- **Decisions for Pedro** (IMPR's recommendation in brackets):
  - **D3 gates:** gate adoption (the ext1 branch of `blp_ext1_adopt.sh`) and K2's `read_alpha` on the FD check (T1/T2). Report stop_reason and moved, but do not gate on them. [Agree with the agent; E3's α is unaffected.]
  - **TOL_OUTER → factr:** [leave it; mapping it would loosen the stop to a 1e-6 relative change].
  - **D8** (δ\* starts for ext2 and extended): [leave it for now; b3's 10× first-evaluation cap covers part of it].
  - **D0 in full, or i+iii+iv:** [in full. The scan (ii) measures how far from θ₀ the contraction starts to fail, which is what lite's r0 = 0.02 assumes. It costs about 1 GPU-h more per routine.]
  - **R0:** [yes, before any promotion. It is the only check of the default path on the GPU.]
  - **Promoting the edited engine into `scripts/`:** [after R0 reproduces 11-06. The flags stay off until the validation passes.]

**5. Running:**
- **K1 `--lambda`:** the script is being edited (`k1_netflow_nls.py` is now `c32b894d`); its tests are pending.
- **The advertising test** and **net-flow Part A.**
- The heavy slot is free. The long runs go one at a time, as each agent reports READY.

## Update 11-11 22:50: status (nothing new from you since 21:25)

- **Your sections:** every one up to 21:25 is answered below. There is no newer section.
- **Scratch cleanup:** not run yet. At 22:45 the targets were still present. The script is unchanged (`9d1f9930…`); Pedro runs it.
- **IF.data register:** still HTTP 500 at 22:46. The `Sr` re-fetch waits.
- **Item 7, BLP engine (D0 + edit b).** This is from the agent's own log; its report and final md5s will follow.
  - **New flags**, all off by default, plus b1, which records the real stop reason:
    - `BLP_OUTER_STEP` off/lite/full (b2);
    - `BLP_INNER_GUARD` (b3);
    - `BLP_INNER_CLAMP_TEST` (b4).
  - **Toy tests:** 31 of 31 pass (`diagnostics/test_solve_box.jl`), after the agent fixed b2's "full" mode at 22:06.
  - **CPU smoke on real data** (two quarters of the E3 frame, 35,381 rows, R = 50): A, B and C pass; D is running.
    - **(A), with b1 on (the default):** it reproduces today's iterates bit for bit, and the start freezes as in production.
    - **The record now names the freeze:** stop "NORM_OF_PROJECTED_GRADIENT_<=_PGTOL" with a line-search "STP = STPMIN" warning, and Q at the end point 224.33 (computed from the polluted δ) against the clean 0.6270 at the start. So even "converged" can label a freeze.
    - **(B), the guard (b3):** bit-identical to today when every contraction converges. A forced failure returns a finite penalty Q and restores δ bit for bit.
  - **Repo files touched so far** (LF md5; they may still change before the report):
    - `blp_engine_cpu.jl` `2b467eb6` → `7d34eaa7`;
    - `blp_engine_gpu.jl` `16261cde` → `c5b9c692`;
    - `blp_ext1_starts.sh` `75ad5510` → `f6920e09` (its `--report` only);
    - new: `blp_diag_ext1_d0.jl` (`02c8d723`), `blp_diag_ext1_d0_job.sh` (`87ea35a9`), `blp_ext1_starts_val1112.csv` (`3f638a06`), and optionally `blp_ext2_starts_r0def1112.csv`;
    - local only: `diagnostics/test_solve_box.jl` (`c2cb9dbe`) and `diagnostics/smoke_engine_flags.jl`.
  - **On Bouchet:** the edited engine goes to a staging folder, not `scripts/`. Production keeps today's engine until Pedro promotes the edit.
  - **Cluster lines prepared, none run:** D0, then 12 validation fits (E3/E4 × control, warm_rc4, cold, rand1–3, flags on), and an optional R0 check that the defaults reproduce today's engine.
- **K1 `--lambda`, the advertising test, net-flow Part A:** the code is being written; no repo file has changed yet (`k1_netflow_nls.py` is still `99eeee52`; `make_paper_numbers.py` is still `4af89d6f`). Their long runs wait until the BLP agent's Julia smoke finishes, then go one at a time.

**This file supersedes `IMPR_REPLY_2026-11-07a.md`.** It answers every 11-11 section of `MAIN_REPLY_2026-11-05.md`, up to and including 21:25. It also carries the full result tables of 11-08 and 11-09 (section 6), so 07a could be retired without losing anything. It was deleted on 11-12 at Pedro's request; its history is in the log of `SHARED_MEMORY.md`. IMPR deletes nothing in `claude/`.

## Update 11-11 22:35: Pedro's go on four items; three small edits done; three runs started

**Pedro (22:2x, to IMPR directly):**
- the K1 `--lambda` edit and the 50-market smoke;
- the advertising test's full run (A1), with both contrasts (A2), the 24-conglomerate pool (A3) and the CR2 p for the five largest (A6);
- the net-flow Part A now, and Part B's Stage 1 after the BLP engine agent, with the Findex variant f (N1–N4);
- the small edits C.3, C.4 and N10.

**Done by IMPR (22:24–22:31). Backups are in `scratchpad\edits_1111\`.**
- **C.4, two new keys:**
  - **`demand.phicap.cap` = 0.99 and `demand.sharecap.cap` = 0.95**, both status ok. The text can quote `\pn{demand.phicap.cap}` and `\pn{demand.sharecap.cap}`.
  - **The checks:** on E3 and E4, max |1 − `_1mphi` − min(phi_mt, 0.99)| over B rows = 0, and the largest B market-quarter Σ share = 0.95 (to 4e-16).
  - **The guard:** if an environment override is set, or a frame was built at another cap, the collector writes a note and builds no key, so the paper prints `??`. That is how this file handles a failed guard. The plan's "mismatch" status does not exist in it.
  - **`make_paper_numbers.py`:** `f51069d8` → **`4af89d6fbb9d077ff8297176b2da575a`** (CR 0).
  - **The rebuild (22:30):** rc 0, **0 MISMATCH**; 414 → 416 entries, ok 337; **no existing value changed** (diffed against a snapshot).
- **C.3, comments only:**
  - `sleep_demand_prep_link.py`: `1e48b66a` → **`27a7db3b5dba0823d874d9eff8e7733c`** (CR 0).
  - The three-line comment at l.83–85 becomes the version without numbers. The comment at about l.560 now says that the floor bounds Σ s at TOTAL_SHARE_CAP.
  - The cluster's copy now differs only in comments; it goes up with the next code upload.
- **N10, the K2 pin:**
  - `diagnostics/diag_k2_netflow_simulation.py`: `1cc8468f` → **`c3c5e41c73e4b3c9261157b45bd07995`** (CRLF kept, 991 CR).
  - `DELTA_BIN_PANEL_MD5` = "fa38794d", plus two docstring lines: the binary is the I2 prep of 11-06; the panel is fa38794d.
  - **Checked:** `read_menu()` now records `delta_source` "binary", 628,612 rows, and the binary equals the closed form to 1.4e-14.
  - No K2 rerun; no number changes.

**Running.** Three max-effort agents, all local. IMPR runs the heavy jobs one at a time; the BLP engine agent's Julia tests hold the machine now.
- **K1 `--lambda`:**
  - **(a)** S-stmt must be bitwise identical to today, reproduce a stored pilot number, and pass test 1 again.
  - **(b)** Then the 50-market R-q smoke, with κ̃₄ and E_c estimated.
  - Outputs go to scratch; `Rout/k1_pilots_fa38794d` is not touched.
- **The advertising test:**
  - a new `diagnostics/diag_p4_adv_group_diff.py` (five gates), producing `pix.adv.diff.*`, `pix.adv.diff_int.*` and `pix.adv.big5.elast_p_cr2`;
  - its `collect_pix_adv` edit is the only change to `make_paper_numbers.py` until it reports.
- **Net-flow Part A:**
  - `diagnostics/diag_netflow_gap.py` → `Rout/diag_netflow_gap/`; gate G-A1 reproduces `k2_numbers.json` first.
  - Then Part B's Stage 1 is prepared: `diagnostics/diag_cap_sensitivity.py`, with frames in `C:\Users\pedro\AppData\Local\Temp\capsens_1111\`.
  - Its 12 local logits run after the BLP engine agent.

## 0. Where your 11-11 requests stand (22:50)

| your section | request | state |
|---|---|---|
| 16:30, item 1 (+ 21:25, point 5) | scratch-removal script | **ready: md5 unchanged, and no folder created since 20:10 is a target.** Pedro runs it (1.1) |
| 16:30, item 2 (+ 20:55, D9) | BLP engine plan (item 7) | **plan done.** D0 and edit (b) are being written locally; D9 checked against the record (1.2) |
| 18:25 to 18:38 (+ 21:20, 21:25) | registry 025399, λ by segment | **done. Pedro's 21:25 decisions applied (S5 fill); the S2 check is done; the re-fetch waits** because IF.data still returns HTTP 500 (1.3) |
| 19:44, both sections | net-flow gap (A–C); caps p1–p6, s1–s3 | **plan done; Part A running; Stage 1 next** (Pedro, 22:2x). C.3 and C.4 done; N5–N7 open (1.4) |
| 19:55 | K2 inputs | **done** (1.5) |
| 20:55 | advertising elasticity, five largest against the rest | **plan done; full run going** (A1–A3, A6). Yes, five clusters break the bootstrap; randomization inference is used (1.6) |
| 19:55, 20:55 | plan renamed `paper_edits_2026-11-11.tex`; α^P to the appendix; `tab:k2b_pairs` in 7.3 | noted |

**Four findings that touch the text:**

1. **The market size never uses the Findex series.** This was found by the 19:44 plan agent; IMPR verified the code path in all four preps.
   - `process_specification`'s `keep_cols` drops `findex_banked_frac`: `sleep_demand_prep_link.py` l.490, and the same line in `sleep_demand_prep_e1.py` l.317 and `sleep_demand_prep_e2.py` l.313.
   - So `build_market_size_and_shares` takes `FINDEX_FALLBACK` = 0.77 for every row (l.356–357).
   - **The text describes the series instead:**
     - V_Friction l.1014–1016 (%AE6) defines bc = 1/(findex_t·access_mt) with findex_t the Global Findex series;
     - l.906 (%AD7) says the Findex "enters the market-size correction".
   - **And %AE7 is wrong under either choice.** It says "When Findex data is unavailable, we fall back to bc = 1.1", but 1.1 is the fallback of the legacy rule only. Under the default rule the fallback is findex = 0.77, and today that fallback applies to every row.
   - The series runs from 0.694 (2016) to 0.864 (2024), so with the series the market sizes would be ×1.11 in 2016 and ×0.89 in 2024.
   - **Pedro decides (N7, section 2): fix and rebuild, or keep 0.77 and say so.**
2. **K2's mean utilities are the I2 logit's, but the run computed them itself and never read the binary.**
   - A stale pin made the run take the closed form. That closed form equals the I2 logit binary to 1.4e-14.
   - "The logit deltas of E3" is therefore numerically exact (1.5).
3. **The S2 rise over the launch window is a four-month drift, not a one-month step.** It moves money from N to S within S2, with no change in the segment's size (1.3).
4. **The advertising difference has been estimated: P8 has it.**
   - Rest minus five largest is +0.019, wild bootstrap p 0.598, with 5 and 63 clusters. So "has not been tested" (VF l.842) is inaccurate.
   - With five clusters on one side, neither the bootstrap nor CR1 is reliable. Randomization inference is the proposed fix.
   - The pooled contrast cannot support "consistent with full awareness" under any method (1.6).

## 1. Answers

### 1.1 The scratch-removal script (16:30 item 1; your 21:25 point 5)

- **The md5 holds:** `9d1f9930c5d6bcf9ffc8912dfe7210bb` at 21:30 (LF, 20,435 bytes, pure ASCII).
- **No folder created since 20:10 is a target.**
  - The 123 targets are explicit paths, with no wildcard. The IMPR-scratchpad targets are lines 56–106 of the script.
  - None of them is, or contains, `blp_engine_impl_1111`, `k2_delta_check_1111`, `lam`, `netflow_plan_1111` or `adv_diff_plan_1111`. The same holds for tonight's `lam_1111` and `s2_check_1111`. The script's own keep list also covers `*_1111.py` and the script itself.
- **What it removes:** 123 targets, 1,663.6 MiB in all.
  - **410.4 MiB under the new root** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\` (57 targets):
    - 375.5 MiB of IMPR scratch, such as the never-submitted K1 array package, the α −0.117 K2 context, the d01b8bae K0 backup, the tables workspace, snapshots and logs;
    - 34.8 MiB of the K2 session 904ee55d.
  - **1,253.2 MiB under the older root** `…\Temp\claude\c--Users-pedro-…-Rep\`:
    - IMPR scratch up to 11-04: `fr2`, `rebuild_smoke`, `e4g_sizing`, `e4g_upload`, `upload_fr3`, `d10`, snapshots and logs;
    - the K2 session's three d01b8bae context pickles;
    - the BB session's data caches. Its 16 Caixa files are byte-identical to the copies in DATA_RAW.
- **What it keeps.** The script header gives the reason for each:
  - `rebuild_1104`, which I10 needs;
  - the `kappa_1105` and `kappa_1109` contexts, for K1 `--lambda` and test 1;
  - `refresh_i2_1106`, `gfb_1108\before_R3` and `ssel_1108`;
  - in the older root, the `i6`–`i9bb` chains, the provenance of `rate_fallback_audit.csv`;
  - the exact-IFT drafts;
  - BB `bd1` and `b7`, and every BB script;
  - 904ee55d `panel_52ea6495`;
  - every `tasks\` folder.
  - The Pix session (77c31b25) is out of scope.
- **Safety:**
  - No target holds a link or junction; the scan did not follow links.
  - Every path goes through the `\\?\` prefix, with a `cmd rd /s /q` fallback. Two paths reach 267 characters.
  - Anything outside the two roots is refused.
- **To run** (PowerShell): first the dry run, which removes nothing, then the real run:
  ```
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\cleanup_scratch_1111.ps1" -DryRun
  ```
  ```
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\cleanup_scratch_1111.ps1"
  ```
  IMPR checks afterwards that every target is gone and every kept folder is still there.

### 1.2 The BLP engine (item 7): plan done, implementation running

- **The plan:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\blp_engine_plan_1111.md` (495 lines, md5 `b34705597d9135b9a259f6589535dbc8`). The update of 11-11 23:14 (point 4) covers the same ground. Its core:
  - **L-BFGS-B's `converged` only means "fewer than 500 evaluations".** The outer tolerance never reaches L-BFGS-B.
  - **A failed inner contraction writes its δ back** (blp_engine_gpu.jl l.1590) and still returns an ordinary-looking Q.
  - **The starts:** across the four routines, 15 of the 16 non-warm starts ended where they began, all reporting `converged = true`. E4's rand2 moved, to Q 63.94.
  - **The cost:** 87 % of ext1's GPU time went to starts that never moved.
- **Your D9 wording (block B1) matches the record.** "All but one of the optimizer's other starting values did not move …, and the one that moved ended at a far higher value of the objective": 15 of 16 did not move, and E4's rand2 ended at Q 63.94 against the chosen 0.036171.
- **Pedro's decisions (20:33):**
  - D0 plus engine edit (b), behind flags whose defaults keep today's behaviour;
  - b1 (the real stop reason) is on by default only if it reproduces today's iterates bit for bit;
  - E3 and E4 first; rand1–3 kept;
  - no Bouchet tonight.
- **The implementation** has been running since 20:35: a max-effort agent, local only.
  - It is editing `blp_engine_cpu.jl` and `blp_engine_gpu.jl` behind those flags, and adds `diagnostics/test_solve_box.jl` (the toy tests).
  - When it reports, the following will be posted here: the md5s, the check that the defaults are unchanged, the test record, the upload list and the launch lines, which will not be run.
- **Still open for Pedro:** D3's gates, TOL_OUTER → factr, D4, D6, D7 and D8 (section 2).

### 1.3 Registry 025399, λ by segment (18:25–18:38; your 21:20 and 21:25)

**Pedro's 21:25 decisions, applied at 21:38–21:50:**

- **S5 fill: the first full-stock value, 2019-03, held back.**
  - `make_lambda_registry.py`: `af16b1f86f115ce81bf6f31d69e7a132` → **`cd2d60d7f9fa396460cea0cba5e14acc`** (CR 0). The backup is `scratchpad\lam_1111\make_lambda_registry_af16b1f8.py`. Rerun rc 0.
  - S5's months before 2019-03 no longer enter its quarterly or December series:
    - R-q holds back the 2019-03 value;
    - R-const for S5 averages 2019Q1–2024Q4;
    - R-annual holds back December 2019 for 2016–2019.
  - **Moot for B, as checked against a before-copy of all 12 outputs:**
    - every B column of `lambda_conglomerate_quarter.csv` is identical;
    - S5 has 3,071 conglomerate-quarters, none of them B;
    - the B-weighted means by year, the correlations, the segment map and the B segment paths are identical.
  - **What changed, S5 only:**
    - R-q 2016–2018 (2018Q4: 0.065 → 0.070);
    - R-annual 2016–2019 (→ 0.128, from December 2019);
    - R-const (0.155 at 3 dp).
    - The jump report loses S5's two phase-in jumps: 11 rows → 9, of which 5 are PF/PJ.
  - **The four keys are unchanged.** They read the monthly shares as delivered: 0.868, 0.924, 0.065 and 0.799.
  - `make_paper_numbers.py` was rebuilt at 21:53 (rc 0, **0 MISMATCH**; ok 335, pending-_ms983 66, pending-_sc983 12, dropped 1). It was diffed against a before-snapshot: **0 of 414 entries changed.**
- **A point on `reg.lambda.pfpj.all.min` (your call).**
  - Its minimum, 0.065, is S5 in 2018-12. That month is inside S5's phase-in, which the series now leave out.
  - Over the same Decembers without the phase-in, the minimum is **0.106 (S4, 2023-12)**.
  - Say whether the key should leave out the phase-in. The PF/PJ December shares:

  | December | S1 | S2 | S3 | S4 | S5 |
  |---|---|---|---|---|---|
  | 2018 | 0.898 | 0.640 | 0.307 | 0.210 | 0.065 (phase-in) |
  | 2019 | 0.908 | 0.491 | 0.284 | 0.223 | 0.128 |
  | 2020 | 0.924 | 0.512 | 0.324 | 0.287 | 0.167 |
  | 2021 | 0.906 | 0.787 | 0.275 | 0.341 | 0.181 |
  | 2022 | 0.868 | 0.821 | 0.304 | 0.136 | 0.185 |
  | 2023 | 0.893 | 0.802 | 0.403 | 0.106 | 0.158 |
  | 2024 | 0.879 | 0.734 | 0.447 | 0.118 | 0.150 |

- **The three rules, one line each** (shares of B time deposits, dep_a4, 2016–2024):
  - **Leader route** (the registry reports by prudential conglomerate: PAN under BTG from 2021Q2, Modal under XP): 49 B rows (38 with deposits), **0.06 %**.
  - **R-fixmem's last panel quarter for conglomerates that leave before 2019Q4:** 2 B conglomerates (Kirton Bank, last quarter 2016Q3, S1; Banco Neon, 2018Q1, S4), 12 B rows, **0.26 %**.
    - **Correction to the 21:00 section:** its "13 conglomerates" counted D conglomerates too (11 D + 2 B). The R versions apply to B only.
    - The 6 B conglomerates that enter after 2019Q4 take their first panel quarter: 85 rows, 0.19 %.
  - **R-annual's 2016–2017 fill under the first-value rule:** 499 B rows, **10.82 %** under the fill. That includes the 313 rows before 2017Q2 (6.18 %) that every R version fills, because the segmentation starts in 2017Q2.
- **The S2 rise, a read-only check from the archived workbook** (md5 `d2eeb450…`). Files: `scratchpad\s2_check_1111\s2_step_drift.py` and `.out`.
  - **PF against PJ cannot be done.** The workbook's holder field has one combined "PF/PJ" category (the others: institutional investor, non-bank FI, bank, non-resident). The check is by holder type and by month.
  - **It is a drift over four months, not a one-month step.**
    - The PF/PJ S share in S2 is 0.441 (2020-09), 0.512 (2020-12) and 0.683 (2021-03), a rise of +0.242.
    - The monthly changes from October: −0.010, −0.000, **+0.081 (Dec)**, +0.068, +0.048, +0.056.
    - The largest month is 33 % of the rise.
  - **It is a re-composition, not a change in the segment's size.**
    - The S stock goes from R$ 61.4 bn to 105.0 bn while the N stock goes from 78.0 bn to 48.8 bn.
    - The total stays between R$ 139 and 154 bn, so nothing suggests an institution entering S2.
  - **It continues after the window:** 0.740 at 2021-06 and 0.787 at 2021-12.
    - Before it, the share had fallen from 0.506 (2020-01) to 0.431 (2020-11), while the N stock grew about 9 % a month in May–July 2020.
  - **Other holders.** The institutional investors' S share in S2 *fell* over the same window (0.648 → 0.182, on R$ 1–2 bn). Whatever moved PF/PJ did not move every holder alike. This is weak evidence against a register-wide recoding, given that small stock.
  - **Reading.** The data rule out a recording step in one month. They cannot tell behaviour from a reclassification applied as CDBs roll over. As decided, every R-q result on the launch change goes next to R-const, and a gap is attributed to S2 first.
- **Re-fetch of the IF.data register:** probed at 21:50, still HTTP 500. IMPR probes again later. When the service answers, IMPR re-fetches the 36 quarters read-only into a new dated folder and reports any `Sr` difference.
- **NOTES.md:** the pasted answer text is noted (NOTES.md md5 `443cf7d08dd9d9b969d2282d77342dd2`).
  - Our convention needs one row, because row 404 says "Answer text to be pasted into NOTES.md". So one row was appended (append-only) to:
    - the master `DATA_RAW\source_log.csv`: 412 → 413 data rows, prefix byte-identical, md5 `ab6463b3d75476ad80bdcc668203f501`;
    - the folder's `source_log_rows.csv`: md5 `a1825b9ba6a66bae3cc2e8296f5fd520`.
  - The row records the paste and that there is no BCB follow-up.
- **K1 `--lambda` and the 50-market smoke: waiting for Pedro's direct go.** IMPR asked him in this session.
  - **The design:**
    - an option `--lambda {S-stmt, R-q, R-const, R-fixmem, R-annual}`, default S-stmt, mutually exclusive with `--lambda-one`;
    - λ read by conglomerate and quarter for B; a missing value stops the run (0 are missing);
    - a guard on the panel md5;
    - the version recorded in `k1_numbers.json`, with outputs in `Rout/k1_<version>`;
    - S-stmt must reproduce today's tables exactly.
  - **The R versions estimate κ̃₄ and E_c, as pilots, under K1's caveat** (Pedro, 21:25).

**The registry results of 21:00, still current** (only the S5 fill and the rule lines above are new):

| key | value | detail | prints `\pn[2]` |
|---|---|---|---|
| `reg.lambda.pfpj.S1.min` | 0.868 | 2022-12 | 0.87 |
| `reg.lambda.pfpj.S1.max` | 0.924 | 2020-12 | 0.92 |
| `reg.lambda.pfpj.all.min` | 0.065 | S5, 2018-12 (see the point above) | 0.07 |
| `reg.lambda.B.tdw.2019Q4` | 0.799 | 61 B conglomerates, 56 with dep_a4 > 0; weights S1 .792, S2 .108, S3 .070, S4 .030 | 0.80 |

- **Archive** (`DATA_RAW\BCB\registry_025399\`):
  - the workbook in `raw\` (md5 `d2eeb450…`, equal to the Downloads copy);
  - NOTES.md, now with the answer text;
  - two B3 documents in `docs\`.
  - The Parecer 1461/2026 was already archived (source-log row 286).
- **RESGATE:** Pedro's reading stands (S = can be redeemed before maturity, an upper bound on daily liquidity). Two points are unsettled by public documentation: whether M is folded into S, and whether layouts before 2024 had only S and N. **No BCB follow-up** (Pedro, 21:25).
- **The segment map:**
  - **Source:** IF.data register field `Sr`. The copy used is byte-identical to the pipeline's 2026-04-07 download (36 quarterly files, 2016-03 to 2024-12).
  - **Check:** the panel's own `segment` column agrees in all 7,437 filled conglomerate-quarters.
  - **B in 2019Q4, by segment:**
    - S1: 5 conglomerates (Bradesco, Itaú, Santander, Caixa, BB), 79.2 % of B time deposits;
    - S2: 4 (Banrisul, Safra, Citi, BNB), 10.8 %;
    - S3: 18, 7.0 %;
    - S4: 34, 3.0 %;
    - no B conglomerate is ever in S5.
  - **Changes:** 13 of 69 B conglomerates change segment (3.37 % of B time deposits), the largest being XP and Master. No S1 bank ever changes.
- **The five λ series** (`Rout\lambda_registry\`, 12 files with an md5 manifest, LF). The B time-deposit-weighted mean λ (PF/PJ), unchanged by the S5 fill:

| version | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2020Q3→2021Q1 |
|---|---|---|---|---|---|---|---|---|---|---|
| R-q (headline) | .805 | .828 | .804 | .809 | .818 | .827 | .796 | .823 | .819 | +.018 |
| R-const | .803 | .808 | .817 | .818 | .820 | .810 | .804 | .809 | .813 | −.006 |
| R-fixmem | .805 | .828 | .804 | .809 | .818 | .828 | .792 | .814 | .798 | +.019 |
| R-annual | .849 | .853 | .862 | .811 | .806 | .814 | .818 | .793 | .827 | +.013 |
| S-stmt (today's K1) | .0002 | .0002 | .0001 | .0001 | .0001 | .00005 | .00004 | .00003 | .00003 | ≈0 |

- **The launch-window rise in R-q** (+.018) comes from S2, with no membership change; see the S2 check above.
- **Correlations at the B-row level:** .92–.96 among the R versions (.94–.97 weighted).
- **PF/PJ segment-share jumps above 0.10, now five:**
  - 2022Q1, two, explained by Nubank's move from S4 to S3;
  - 2022Q3, probably explained by C6;
  - the S2 launch rise;
  - one S3 move in 2023Q1, with no membership change.

### 1.4 The net-flow gap and the two caps (your two 19:44 sections): plan done

**The plan:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\netflow_gap_cap_plan_1111.md` (594 lines, md5 `218b9d9380a0022ed0396711fd3c2cba`).
- The helpers and their outputs are in `scratchpad\netflow_plan_1111\`, with md5s listed in the plan.
- It was written by a max-effort subagent, and every claim is tagged [V-code], [V-data] or [inf].
- Nothing in production was written and nothing ran on Bouchet.
- Parts A and B are plans. Part C.2 and a "dry forecast" were computed read-only: the dry forecast runs the production `build_market_size_and_shares` on the stored I2 frames with each variant's caps.
- The inputs are `DEMAND_PREP\demand_3_index_spec_12.parquet` (E3) and `demand_4_index_time_spec_12.parquet` (E4), both of 11-06 16:53, on panel fa38794d.

**C.1: confirmed, with three additions.**
- The capped φ enters only through `_1mphi` = 1 − min(φ, 0.99) (l.354–355). That term is used in:
  - the anchor r̂ (l.400–417);
  - the market-size floor (l.434–438, 453–457);
  - the share denominators (l.462–465).
- The active balance (eq. 11-B, the code's "(14-B)") and its truncation use the uncapped φ̂ (l.529–544).
- On the stored frames the match is exact (max difference 0, E3 and E4).
- **Addition (a):** it is the floor, which also uses the capped φ, that keeps Σ s ≤ 0.95, not the anchor.
- **Addition (b):** the cap also binds the D rows' national φ_t, in 20 (E3) and 14 (E4) of 36 quarters.
- **Addition (c):** the frames store the *uncapped* φ̂ (`phi_mt`), and the consumers use three different ceilings: none in the K2 world, 0.999 in CF/BBL, and 0.99 in the shares.

**C.2: φ̂ over B demand rows** (uncapped, spec 12):

| | E3 | E4 |
|---|---|---|
| B rows / B market-quarters | 625,677 / 111,146 | 640,086 / 111,388 |
| p10 / p50 | 0.9562 / 0.9932 | 0.9478 / 0.9909 |
| p90, p99, p99.9 | 1 (p90 = 1 − 1.1e-16; p99 and p99.9 exactly 1) | the same |
| share ≥ 0.98 | 0.706 | 0.583 |
| **share ≥ 0.99 (at the cap)** | **0.542** | **0.506** |
| share ≥ 0.999 / ≥ 0.9999 / ≥ 0.99999 | 0.355 / 0.253 / 0.208 | 0.400 / 0.353 / 0.333 |
| share = 1 | 0.082 | 0.017 |
| **balance-weighted** share ≥ 0.99 / ≥ 0.999 / ≥ 0.9999 / = 1 | **0.235** / 0.080 / 0.045 / 0.012 | **0.179** / 0.103 / 0.076 / 0.0023 |
| market-quarter level: p50; share ≥ 0.99 / ≥ 0.999 / = 1 | 0.9979; 0.637 / 0.442 / 0.105 | 0.9991; 0.615 / 0.502 / 0.022 |

- **For the text:** "about half of the B demand rows" holds by rows (54 % / 51 %), but by balance it is 24 % / 18 %.
- E3's share at the cap by year: 0.28 (2016), 0.36, 0.58, 0.62, 0.77 (2020), 0.71, 0.51, 0.46, 0.56 (2024).
- **64 % (E3) and 69 % (E4) of the market-quarters at or above the q95 anchor are capped**, so the cap shapes r̂.
- **Cross-checks:** the market-quarter quantiles equal K0's `a_phi_quantiles`, and the floor share (4,043 of 111,146) equals K0's `share_paper_floor_binds`.

**C.3: the comment at l.85** ("0.99 barely binds: p90(φ)=0.977") is stale, and so are l.83–84 and l.555–557.
- The recommended replacement has no vintage numbers (plan C.3). It says:
  - the cap binds on about half of the B rows and in most of the q95 anchor's top tail;
  - it binds the D rows' φ_t in some quarters;
  - the `demand.phicap.*` keys measure it;
  - Dep_Act and its truncation use φ uncapped.
- A comment-only edit, on Pedro's go (N8).

**C.4: the keys `demand.phicap.cap` (0.99) and `demand.sharecap.cap` (0.95).**
- They go into the existing collectors `collect_phi_cap` and `collect_share_cap`, once each.
- Two guards:
  - status "mismatch" if `DEMAND_PHI_CAP` or `DEMAND_TOTAL_SHARE_CAP` is set in the environment;
  - a check against each routine's newest spec-12 parquet: the max of 1 − `_1mphi` equals the cap to 1e-12, and the max Σ `share_B_cond` equals the share cap to 1e-9.
- On Pedro's go (N9).

**Part A: four diagnostics, one new local script** (`diagnostics/diag_netflow_gap.py`, read-only, about 20 minutes, outputs in `Rout/diag_netflow_gap/`):
- **A.1, the world's own net flows.**
  - The stored K2 world keeps only summaries, so A.1 rebuilds it with K2's own functions. **Gate:** the rebuild must reproduce `k2_numbers.json` to 1e-12 (the awake share 0.0209676, the stay probability 0.1406020, the moving-share quantiles).
  - It then reports |net flow| / carried stock by cell, next to the data's. For k4: p25 0.042, p50 0.102, p75 0.216, p90 0.425.
  - The worlds:
    - **W0**, the K2 world, with the paper's 2.1 % and 14 %;
    - **W1**, κ 0 with the stored δ, which equals the data on on-menu cells with φ̂ ≤ 0.99;
    - **W2**, awake money from the capped φ, the world consistent with the demand side;
    - **W3**, κ 0 with the cap;
    - **W4**, the statements' ω for maturities. W4 needs an ω-aware copy of `simulate` inside the new script, since K2 itself is not edited.
- **The plan agent's expectation [inf]: much of the gap is built into the world, not into behaviour.**
  - The K2 world takes awake money from the uncapped φ̂, while its δ come from shares built with the capped φ.
  - So in capped cells (56 % of type-4 K0 cells) the world's inflow is scaled by (1 − φ̂)/(1 − 0.99), and it is zero where φ̂ = 1 (8 %).
  - In truncated cells (41 %) the world can move at most (1 − φ̂)·carry.
- **A.2:** the variance of the net-flow ratio, split into a market-quarter part and a within-market part (carry-weighted, winsorised), by deposit type.
- **A.3:** the share of the top-decile |net flow| cells that coincide with ESTBAN branch changes, conglomerate composition changes, ESTBAN filing gaps or technical flags, against the share among all cells.
- **A.4:** the outflow side at ω = 1 is the same inequality as the (14-B) truncation, so it equals `demand.flows.trunc_share` (E3 0.4546) on a common cell set. The inflow side is zero by construction under the capped φ and the floored M̂, so it is reported against the uncapped φ̂ and the market size before the floor.

**Part B: cap sensitivity, logit first, all local.** The logit needs no cluster: `blp_logit.jl --demand-dir --out-dir` writes nothing to Drafts, and its fixed WCB seed reproduces se exactly.
- **Stage 1, a screen (about 1.5–2 h):**
  - re-share the stored I2 frames under each cap (base\*, p1–p6, s1–s3) and run 11 logits;
  - **gate:** the re-shared base α must match production within 0.05·se (production E3 α −0.064080, se 0.031502).
- **Stage 2, exact (about 40 minutes plus the logits):** reruns the full prep in-process, for the variants to be reported or sent to BLP.
  - Its base must reproduce the production parquet and α.
  - **One risk:** the local pickles are the 11-07 re-ingest, while the frames are from 11-06.
- **p3 and s3 have no existing knob.** They become sandbox-only row filters, with no production edit:
  - **p3a** drops B rows with φ̂ ≥ 0.99 after the shares are formed: 54 % of B rows; outside median 0.834.
  - **p3b** drops them before r̂: r̂ −22 %.
  - **s3** drops the floor market-quarters: 6.5 % of B rows.
- **The first-stage F** (Wald and Montiel Olea–Pflueger) is not in blp_logit's output, so a helper computes it. It is the same for every variant that keeps the base sample; only p3 and s3 change it.
- **The dry forecast** (E3, production function, Findex as in production):

  | variant | caps (φ / share) | r̂ × base (E3 / E4) | B rows at the φ cap | median outside share | median Σ_B s where φ̂ < 0.98 (p10) |
  |---|---|---|---|---|---|
  | base | 0.99 / 0.95 | 1 / 1 | 0.542 | 0.864 | 0.1015 (0.0178) |
  | p1 | 0.98 / 0.95 | 0.65 / 0.66 | 0.706 | 0.872 | 0.1559 (0.0273) |
  | p2 | 0.995 / 0.95 | 1.58 / 1.64 | 0.472 | 0.878 | 0.0643 (0.0113) |
  | p4 | 0.999 / 0.95 | 5.0 / 6.1 | 0.355 | 0.938 | 0.0202 (0.0035) |
  | p5 | 0.9999 / 0.95 | 27.9 / 48.7 | 0.253 | 0.980 | 0.0036 (0.0006) |
  | p6 | 0.99999 / 0.95 | 198.8 / 450.5 | 0.208 | 0.997 | 0.0005 (0.0001) |
  | s1 / s2 | 0.99 / 0.90, 0.98 | 1 / 1 | 0.542 | 0.864 | 0.1015 |

  - **Pedro's 0.99999 pushes ordinary markets to a near-one outside share.**
    - As the cap nears 1, the markets with φ̂ = 1 (8 % of E3 rows) take over the top of the anchor, so r̂ grows like 1/(1 − cap).
    - p5 and p6 are a different market-size model, not "the same model with a looser cap".
  - **p7 would answer his question without that:** take the anchor only over market-quarters with φ̂ below the cap. It needs a new prep knob.
  - The mean ρ(1 − s) moves by at most 1.1 %, so the elasticity moves almost one for one with α.
- **The rule for proposing BLP ext1, to be fixed before anything runs:**
  - **M0 (admissible):** G_t > 0 in all quarters, and the median Σ_B s in markets with φ̂ < 0.98 at least 0.01. p5 and p6 fail M0.
  - **Then any of:**
    - M1, the mean own-price elasticity moves by at least 10 %;
    - M2, |Δα| ≥ 0.5·se (0.0158 for E3, 0.0160 for E4);
    - M3, the 5 % star on α changes.
  - For calibration, TR5 moved the logit α by 9 % and 5 %.
  - **Cost if triggered:** about 0.5–1.5 GPU-h per routine and variant. It also needs an override knob in `blp_ext1_starts.sh`, a code change on Pedro's go.
- **What a different cap would move:** plan B.7 lists every reader of the market size or the capped φ, with line numbers:
  - frames, logit, RC-BLP and ξ̂;
  - CF/BBL, which use M_mt and clamp φ at 0.999;
  - K0's outside pool and `kappa.k0.outside.*`;
  - the K2 world and K2 NLS (`kappa.k2.*`, `kappa.k2b.*`);
  - K1;
  - the cap keys; TR5; two diagnostics.
- **Two more flags, with no production effect:**
  - `diag_k0_netflow_objects.py:165` hard-codes the floor check at 0.95, and `diag_k2_netflow_simulation.py:897` hard-codes its check at 0.99. Under the s- or p-variants they would mislabel.

### 1.5 K2 inputs (19:55): done

Done by a max-effort subagent: read-only, no repo file written, K2 not rerun. Files: `scratchpad\k2_delta_check_1111\` (`compare_deltas.py`, `.json`, `dump_ext1_delta.jl`). IMPR re-checked the pin and the rule in the code.

**1. The 11-09 a216 run did not read the binary. It computed the same numbers itself.**
- **Both records say so.** `Rout\k2_fa38794d_a216\k2_nls_numbers.json` and `netflow_sandbox\Rout\k2_numbers.json` record:
  - `delta_source` "closed form";
  - `delta_binary_rows` null;
  - `frame_panel_md5` fa38794dcb3bed514358d3c2406f82ab;
  - `delta_binary_panel_md5` d01b8bae;
  - `prep_rows` 628,612.
- **The log agrees.** The chain log printed three times: "frame panel fa38794d, the binary's d01b8bae: delta from the closed form". Steps 5, 6, 7 and 2b all loaded the same context, `kappa_1109\k2n_ctx_fa38794d_a216.pkl`.
- **Why.** `diag_k2_netflow_simulation.py` reads the binary only when the frame's panel md5 starts with `DELTA_BIN_PANEL_MD5`, which l.136 pins at "d01b8bae" (the 10-27 panel). The I2 frame says fa38794d, so the closed form ln s_j − ln s_0 was computed from the frame instead: the formula of `blp_logit.jl`'s `berry_inversion`.
- **The frame:** `DEMAND_PREP\demand_3_index_spec_12.parquet`, md5 `0dd20428ec7e54a6b7bcf57056b0255b`, 628,612 rows, 11-06 16:53, metadata panel fa38794d.
- **The binary on disk is I2's:** `BLP_RESULTS\logit\logit_delta_E3_spec_12.bin`, md5 `7952d49b05de4be27060f71e67b49770`, n = 628,612, written 11-06 16:41 by the I2 ingest. Nothing records its panel.
  - Recomputed read-only, it **equals the closed form on this frame to 1.4e-14** (1.1e-14 on B type 4), well inside the script's 1e-8 gate.
  - The context's type-4 δ equals today's closed form exactly.
- **So "the logit deltas of E3" is numerically exact;** only the provenance record says "closed form".
  - The pin is stale. Setting it to fa38794d, a one-line change, would make future runs read the binary and check it. That is optional and changes no number (N10).

**2. Type-4 mean utilities: ext1's δ\* against the logit's** (ext1 minus logit; "mt" = market-quarter; "after mt" removes each market-quarter's mean difference):
- **Inputs.** ext1's δ\* comes from `cluster_raw\blp_checkpoint_E3_spec_12_ext1.jls` (md5 `ea280468…`; no `.bin` came back from I2). It is identical to the δ in `blp_results_E3_spec_12_ext1.jls`, and its α is −0.2160952.
- **Units.** 180,057 type-4 products, merged onto K2's units as `build()` does.
- **Check.** The logit numbers reproduce K2's record exactly: the within slope of δ on g −0.0468, within sd 0.929, ideal constant −4.690.

| object | corr | corr within mt | mean abs diff | median abs diff | mean diff | share \|Δ\| > 1 | mean abs diff after mt | share \|Δ\| > 1 after mt |
|---|---|---|---|---|---|---|---|---|
| **δ, type 4** | **0.782** | 0.716 | **1.294** | 0.788 | −0.051 | 40.9 % | **0.519** | 13.5 % |
| n = δ + αg, each with its own α (logit −0.064) | 0.790 | 0.726 | 1.417 | 0.916 | −0.370 | 46.5 % | 0.545 | 15.4 % |
| simulated δ, fe world | 0.810 | 0.845 | 1.172 | 0.743 | −0.051 | 37.6 % | 0.312 | 5.4 % |
| simulated δ, stored world | 0.783 | 0.723 | 1.301 | 0.796 | −0.051 | 41.2 % | 0.542 | 14.7 % |

- **Other moments:** the rank correlation is 0.892. The standard deviation is 2.83 for the logit δ and 4.26 for ext1's.
- **The gap is the random coefficients, not a renormalization.**
  - Both are inversions of the same shares with an outside utility of 0.
  - ext1's π terms (spread × demographics, log assets × GDP) shift δ by type and market-quarter, and the shift grows with the spread level.
  - For types 1, 2 and 5, 99.8–99.98 % of the gap's variance is between market-quarters; for type 4 it is only 69 %.
  - Type 4's within-market-quarter gap is largest when the policy rate was high: 0.81–1.05 in 2022–23 and 0.58 in 2016, against 0.18–0.24 in 2019–21.
- **Caveats (the agent's, IMPR agrees):**
  1. K2's world has no random coefficients. In it, the logit δ is the one that reproduces the frame's shares (the κ = 0 check, median 1.0012), while ext1's δ\* reproduces them only with the random coefficients integrated. A swap is not like for like.
  2. Types 1, 2, 5 and D keep their stored δ in every world, and they sit in the share denominators. Under ext1 they would move by large shifts: mean absolute gaps of 3.34 (type 1), 0.91 (type 2), 1.91 (type 5) and 1.31 (D). The type-4 numbers therefore understate how much the simulation would change.
  3. Step 7 adds the extended stage's random coefficients to the logit δ, not to ext1's δ\*.
  - The α that goes with the logit δ is −0.0641 (se 0.0315). K2 pairs the logit δ with ext1's α −0.216, as the paper now says.

### 1.6 The advertising elasticity, five largest against the rest (20:55): plan done

**The plan:** `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\adv_diff_test_plan_1111.md`.
- The helpers are in `scratchpad\adv_diff_plan_1111\`.
- It was written by a max-effort subagent. It is a plan only: no repo script was edited and the test was not run. The checks it made were read-only, on the design and on descriptive statistics.

**1. P8 already estimated the difference; it just never became a key.** IMPR checked the JSON.
- `Rout/p4_openings_advertising.json` (panel fa38794d), `groups_B.adv_big5_minus_rest`: five largest minus rest −0.0188 (se 0.0372). So **rest minus five largest is +0.019**.
  - CR1 95 % interval [−0.054, 0.092].
  - Wild cluster bootstrap p **0.598**, 95 % interval [−0.154, 0.228].
  - Clusters: 5 (179 obs) and 63 (1,771 obs).
- The same file holds two more contrasts:
  - the intensive margin (`intensive_B`): rest minus five largest +0.190 (se 0.070), CR1 p about 0.008, bootstrap p 0.116;
  - the ratio form (`ratio_B`): +0.143, bootstrap p 0.21.
- **So "the difference between the two groups has not been tested" (VF l.842) is inaccurate as the code stands.** The wording is Pedro's (A7).

**2. Your first question: yes, five clusters make the wild cluster bootstrap unreliable here.**
- **The identifying variation is concentrated.** 96.1 % of the variation of advertising × five-largest sits in the five clusters, and 73.7 % in Caixa alone. The effective number of clusters is **1.80 of 68**.
  - Caixa's budget follows a cycle, low in Q1 and high in Q4.
  - The other four barely vary (within sd 0.30–0.37).
- **With this few treated clusters,** the restricted bootstrap tends to under-reject and CR1 to over-reject (MacKinnon and Webb 2017, 2018). In P8 the bootstrap interval is 2.6 times CR1's width, and lopsided.
- **For the pooled contrast the verdict does not depend on the method.**
  - CR1 (p 0.615) and the bootstrap (p 0.598) both fail to reject.
  - Even CR1's too-narrow interval [−0.054, 0.092] covers both 0 and the rest's own elasticity (0.086). It can rule out neither "the same response as the rest" nor "no response at the five largest".
  - So the pooled contrast cannot support "consistent with full awareness" under any method.
- **The method matters in two places:**
  - the intensive margin: CR1 p 0.008 against bootstrap p 0.116;
  - the five largest's own elasticity: CR1 p 0.005 against bootstrap p 0.097. The full-awareness reading rests on that 0.097.

**3. What to use instead.**
- **Primary: randomization inference on the t statistic** (MacKinnon and Webb 2020), over which five conglomerates carry the label.
  - **The pool:** the 24 B conglomerates that advertise in every sample quarter, with at least 12 quarters. All 42,504 five-sets are enumerated exactly.
  - The same P8 regression and the same CR1 t; the 95 % set comes from test inversion.
  - **In this pool the five largest are typical:** 15th to 18th percentile of identifying variation.
  - **Not the full 60-conglomerate pool.** There the five largest sit at the 0.4th percentile, so they are not exchangeable and the test would over-reject.
- **Robustness: CR2 with Bell–McCaffrey degrees of freedom** (Imbens and Kolesár 2016; Pustejovsky and Tipton 2018). It needs no exchangeability, and is expected to be conservative.
- **Diagnostics, with no keys:** each cluster's share of the identifying variation, the estimate dropping each of the five in turn (Caixa first), and a big-four estimate.
- **Rejected:**
  - a subcluster bootstrap (ln o autocorrelation is 0.52 at lag 4);
  - Conley–Taber (error sd 0.25–0.48 for the big four against 0.74 for the rest);
  - sign-change tests (with five clusters the smallest possible p is 0.0625).

**4. The specification is unchanged from P8 g1** (`diag_p4_openings_advertising.py:475`): ln o = a_j + t_q + βx + δ·x·1[j∈S] + pρ_{t−1}, with x = ln(1 + adv). n 1,950, G 68. Rest minus five largest = −δ.

**5. Keys.**
- **Available now from the P8 JSON, with no run:**
  - `pix.adv.diff.elast` 0.019 and `pix.adv.diff.p` 0.598;
  - `pix.adv.diff.G.big5` 5 and `pix.adv.diff.G.rest` 63;
  - optionally `.ci.lo` and `.ci.hi` (−0.154, 0.228).
- **With the new run:**
  - `.p_ri`, `.ri.lo` and `.ri.hi`, `.ri_n` (42,504) and `.ri_pool` (24);
  - `.p_cr2` and `.df_cr2`;
  - design only: `.G_eff` (1.80) and `.caixa_share` (0.737);
  - optional: `pix.adv.diff_int.*` (the intensive margin) and `pix.adv.big5.elast_p_cr2`.

**6. Code and cost.**
- **A new script, `diagnostics/diag_p4_adv_group_diff.py`.** It imports P4's `design`, `Fit` and `run`, reads only P4's CSV and JSON and P2's coverage file (no panel), and writes `Rout/p4_adv_group_diff.json`.
- **Five gates before anything is written:**
  - the panel md5;
  - n and G;
  - its estimate and CR1 se match P4's;
  - an exact bootstrap replay reproduces 0.598 and [−0.2282, 0.1536];
  - the CR2 unit tests pass. No R or clubSandwich is on this machine to cross-check them.
- **One edit to an existing file:** `make_paper_numbers.collect_pix_adv`, about 25 lines. The keys appear only when the sidecar's md5s match.
- **About 2–5 minutes, locally.**
- **Citations:** MacKinnon–Webb 2018 and 2020, Pustejovsky–Tipton 2018 and Canay–Santos–Shaikh 2021 are not in References.bib. Randomization inference in the text would need a MacKinnon–Webb 2020 entry (bib work, on Pedro's go).

## 2. Decisions for Pedro (IMPR's recommendation in brackets; items decided at 22:2x are marked)

**The BLP engine (item 7):**
- **D3:** b1 on by default, with adoption and the K2 guard gated on the stop reason plus the FD check [yes].
- **TOL_OUTER → factr** [yes, behind the flag].
- **D4:** the acceptance rule [the plan's: a clean re-evaluated Q to 1e-7, and at least 8 of 11 starts count].
- **D6:** the start design, plus a 13th, cold start [as planned].
- **D7:** adoption only on a lower Q that passes FD and the stop reason [only with Pedro's go].
- **D8:** δ\* warm starts for ext2 and extended [yes].

**K1 and the registry:**
- **The `--lambda` edit of `kappa/k1_netflow_nls.py` and the 50-market smoke:** **decided 22:2x, go** (running).
- **`reg.lambda.pfpj.all.min`:** keep S5's 2018-12 (0.065), or leave out the phase-in (0.106, S4 2023-12). This is the main thread's call.

**The 19:44 plan:**
- **N1. Part A:** run `diag_netflow_gap.py` locally (about 20 min, read-only) [yes; W4 included; E3 only, as K0 and K2 use E3]. **Decided 22:2x, go** (running).
- **N2. Part B, Stage 1 screen** (about 1.5–2 h local) [yes]. IMPR suggested one added variant, **f**: the Findex series instead of 0.77. It costs one more logit (about 6–8 min) and sizes N7 before any rebuild. **Decided 22:2x, go with variant f,** after the BLP agent's runs.
- **N3. p3a or p3b** [p3a, your definition; p3b optional]. **Decided: as recommended.**
- **N4. p5 and p6** [keep them as logit-only evidence that a cap near 1 breaks the anchor]. **p7**, an anchor over φ̂ below the cap, needs a new prep knob [only if Pedro wants a cap near 1 as a live option]. **Decided: p5 and p6 logit-only; no p7.**
- **N5. The BLP trigger rule M0–M3** [approve].
- **N6. BLP ext1 for triggered variants,** plus the override knob in `blp_ext1_starts.sh` [only after N2 shows a trigger].
- **N7. Findex:** fix `keep_cols` and rebuild, or keep 0.77 and say so in the text [decide after N2's variant f]. %AE7's "bc = 1.1" needs a text fix under either choice.
- **N8. The C.3 comment edit** [yes, the version without numbers]. **Done 22:28.**
- **N9. The C.4 keys** [yes]. **Done 22:28.**

**K2:**
- **N10.** Set `DELTA_BIN_PANEL_MD5` to fa38794d, so K2 reads the binary and checks it [yes; it changes no number]. **Done 22:28.**

**The advertising difference (20:55 plan):**
- **A1.** Register only the existing P8 numbers, or also run the new script (local, 2–5 min, read-only on P4's outputs) [the full run: the existing p is the one the plan shows to be unreliable]. **Decided 22:2x: the full run.**
- **A2.** The pooled contrast only, or also the intensive margin, both fixed before the run [both]. **Decided: both.**
- **A3.** The pool for the randomization test: the 24 steady advertisers, all 28 steady advertisers, or the top 15 by 2019 deposits [the 24]. **Decided: the 24.**
- **A4.** Which p the sentence quotes [the randomization p first; the main thread words it].
- **A5.** Whether to say the comparison rests mostly on Caixa. Its openings jump in 2020, 2022 and 2024, which look like government benefit-account waves (an inference, not checked) [the main thread's call].
- **A6.** Add the CR2 p for the five largest's own elasticity [yes, it is cheap]. **Decided: yes.**
- **A7.** The "has not been tested" wording [Pedro and the main thread].

**Housekeeping:**
- Run `cleanup_scratch_1111.ps1`.
- I10's go.
- Bradesco 2016, at the next rebuild.

## 3. State (22:50)

- **I2 (panel `fa38794d`)** is what everything reads. The demand frames are of 11-06. The paper and CF/BBL read ext1. E4's extended stage (e3ext) was adopted on 11-07 at 23:00 (K > L, SE refused).
- **Paper numbers:** last rebuilt at 22:30, after the C.4 keys: rc 0, **0 MISMATCH**; 416 entries (ok 337, pending-_ms983 66, pending-_sc983 12, dropped 1).
  - The 21:53 rebuild changed nothing.
  - The 22:30 rebuild only added `demand.phicap.cap` and `demand.sharecap.cap`.
  - **Nothing is stale:** TR5, gfb, ssel, K2, K2b and K0's `menu_share` are all on I2.
- **The BLP engine files in the repo carry the item-7 edit, behind flags.** The defaults reproduce today's iterates bit for bit (the agent's smoke A). Nothing is on Bouchet.
- **K2 and K2b** at α −0.216 were promoted on 11-09 at 16:31. The K2b gate error is 0.080 against 0.05, reported as a finding.
- **K1:** option (iii); the pilots are in `Rout/k1_pilots_fa38794d`. Test 1 is re-based with `--recovery` and passes (5.6e-17).
- **V_Friction tables:** all current at the 11-09 20:25 check. Since then only macros changed: the registry keys at 20:42, and the two cap keys at 22:30. No existing value changed.

## 4. Script and file md5s (LF)

| file | md5 | state |
|---|---|---|
| `make_lambda_registry.py` | `cd2d60d7f9fa396460cea0cba5e14acc` | **edited 21:38** (S5 fill; from `af16b1f8`) |
| `make_paper_numbers.py` | `4af89d6fbb9d077ff8297176b2da575a` | **edited 22:28** (C.4 keys; from `f51069d8`); the advertising agent edits it next |
| `sleep_demand_prep_link.py` | `27a7db3b5dba0823d874d9eff8e7733c` | **edited 22:28** (C.3 comments only; from `1e48b66a`) |
| `kappa/k1_netflow_nls.py` | `99eeee52d7cd58ac62c2ff34bcc5ca1a` → being edited (`c32b894d` at 23:06) | the K1 agent (`--lambda`); final md5 when it reports |
| `blp_engine_cpu.jl` / `blp_engine_gpu.jl` | `7d34eaa7…` / `c5b9c692…` (from `2b467eb6` / `16261cde`) | **item-7 edit, 23:03:** flags b1–b4; the defaults reproduce today |
| `blp_ext1_starts.sh` | `f6920e09…` (from `75ad5510`) | `--report` only |
| new: `blp_diag_ext1_d0.jl`, `blp_diag_ext1_d0_job.sh`, `blp_ext1_starts_val1112.csv`, `blp_ext2_starts_r0def1112.csv`, `diagnostics/test_solve_box.jl`, `diagnostics/smoke_engine_flags.jl` | `02c8d723`, `87ea35a9`, `3f638a06`, `4881d760`, `c2cb9dbe`, `8dd5a4d0` | item 7; the last two are local only |
| `diagnostics/diag_k2_netflow_simulation.py` | `c3c5e41c73e4b3c9261157b45bd07995` | **edited 22:28** (N10 pin; from `1cc8468f`; CRLF kept) |
| `diagnostics/diag_k2_nls_simulation.py` | `f517111f35b9997065130dd045cfca2b` | unchanged |
| `scratchpad\cleanup_scratch_1111.ps1` | `9d1f9930c5d6bcf9ffc8912dfe7210bb` | run by Pedro, then deleted (23:00); verified 23:05 |
| `DATA_RAW\source_log.csv` | `ab6463b3d75476ad80bdcc668203f501` (raw, CRLF) | one row appended 21:50 |

## 5. Open items

| # | owner | item | state |
|---|---|---|---|
| 7 | Pedro | BLP engine: D0 and edit (b) are done locally (23:03); the cluster lines are prepared, not run | D3, TOL_OUTER, D8, D0's scope, R0 and promotion (23:14, point 4); runs at the next Bouchet session |
| 7b | Pedro | I10, productionization (draft plan `scratchpad\rebuild_1104\I10_plan.md`, about 15–18 h of agent work) | waiting for go |
| 8 | Pedro | Bradesco 2016 (A plus HSBC) | next rebuild |
| 9 | IMPR | B-registry handoff | 11-13 |
| 10 | — | `cleanup_scratch_1111.ps1` | **closed: run by Pedro; verified 23:05** (123 of 123 targets gone, 43 of 43 kept items present) |
| 11 | IMPR | IF.data register re-fetch, `Sr` comparison | waits for the service (HTTP 500 at 21:50) |
| 12 | IMPR | K1 `--lambda` and the 50-market smoke | **go 22:2x**; running |
| 13 | IMPR | `reg.lambda.pfpj.all.min` without S5's phase-in (decided 22:55) | edit after the advertising agent reports; the macro goes 0.07 → 0.11 |
| 14 | Pedro | 19:44 plan: N1–N4 decided (Part A running, Stage 1 next); N5, N6, N7 open; N8 and N9 done | N5–N7 |
| 15 | main thread | the Findex text (VF l.906, 1014–1018; %AE7's 1.1) | with N7 |
| 16 | IMPR | advertising-difference test: A1, A2, A3 and A6 decided, running; A4, A5 and A7 are the main thread's | running |
| 17 | IMPR | N10, the K2 pin | **done 22:28** |
| 18 | IMPR | V_Full: S18 equation macros, and the BBL and policy-function tables against the 11-07 E4 extended change | when Pedro recompiles V_Full |

## 6. Full results, 11-08 and 11-09 (carried from 07a, unchanged)

*Carried verbatim from `IMPR_REPLY_2026-11-07a.md` (deleted 11-12), section "Full results, 11-08 and 11-09". R5's md5 row is as of 11-09; section 4 above has the current md5s.*

**R1. TR5 on I2: every `tr5.*` value** (R3 → I2; the production values are I2's)

| variant, routine | mean φ | logit α | BLP ext1 α (se) | production BLP α | Q | n (share of production) | converged |
|---|---|---|---|---|---|---|---|
| no5, E3 | 0.982 → **0.984** (prod 0.984) | −0.075 → **−0.070** | −0.182 (0.147) → **−0.252 (0.299)** | −0.216 | 0.093 → 0.035 | 580,668 → 569,232 (0.924 → 0.906) | yes |
| no5, E4 | 0.981 → **0.980** (prod 0.979) | −0.068 → **−0.069** | −0.138 (0.160) → **−0.242 (0.352)** | −0.227 | 0.090 → 0.039 | 579,178 → 578,581 (0.901 → 0.900) | yes |
| d5only, E3 | 0.982 → **0.984** (prod 0.984) | −0.067 → **−0.067** | −0.017 (0.105) → **−0.336 (4.682)** | −0.216 | 0.081 → 0.027 | 580,924 → 569,467 (0.924 → 0.906) | yes |
| d5only, E4 | 0.981 → **0.981** (prod 0.979) | −0.060 → **−0.066** | −0.120 (0.152) → **−0.237 (4.554)** | −0.227 | 0.084 → 0.031 | 579,412 → 578,463 (0.901 → 0.900) | yes |

**R2. K1 pilots: the 12 fits behind `kappa.k1.pilot.dk.min/max`** (Rout/k1_pilots_fa38794d; 300 markets drawn, 142 in every fit, 16,701 rows, G 14; none converged)

| form | start | dκ̃ | SSR/yy | stop reason | parameter at its box | note |
|---|---|---|---|---|---|---|
| headline | 0 | −0.788 | 0.218052 | maxit | — | interior |
| headline | 0.5 | −5.244 | 0.189822 | maxit | a | boundary |
| headline | 1 | −0.667 | 0.218061 | maxit | — | |
| headline | 1.5 | −8.808 | 0.191144 | maxit | κ, dκ_chk | |
| matured scale | 0 | **−11.981** | 0.216336 | maxit | a_m | **min** |
| matured scale | 0.5 | **+0.316** | 0.218251 | maxit | — | interior, **max** |
| matured scale | 1 | −6.927 | 0.213753 | no descent step | κ_chk, dκ_chk | |
| matured scale | 1.5 | −6.358 | 0.189648 | maxit | κ_chk, dκ_chk | boundary |
| savings-only | 0 | +0.303 | 0.217351 | no descent step | κ | boundary |
| savings-only | 0.5 | −0.102 | 0.221957 | no descent step | — | |
| savings-only | 1 | **−0.341** | 0.219795 | no descent step | — | interior → `so.*` keys |
| savings-only | 1.5 | −0.422 | 0.221356 | no descent step | — | |

- **Savings-only interior:** dκ̃ −0.341. SE 2.107 is the cluster sandwich (CR1, G 14); the wild-cluster-bootstrap SE is 2.031. The 95 % interval [−3.414, 2.732] is the score wild cluster bootstrap (Webb, symmetric percentile-t, 9,999 draws).
- **Promoted file md5s** (raw / LF):

  | form | estimates | numbers |
  |---|---|---|
  | headline | e58adc46 / dd962411 | 42ab8f3c / e8f7bd12 |
  | matured scale | a9283453 / b226442d | 9032f8b3 / dd892155 |
  | savings-only | 934045f1 / 743384e5 | 3096e45c / 01fc2fc1 |

**R3. K2 on I2, the three vintages side by side.**
- **R3 (d01b8bae):** what the plan was written on.
- **I2, α −0.117:** the 11-08 rerun, now backed up in `Rout/k2_fa38794d_a117_toplevel`.
- **I2, α −0.216:** the 11-09 rerun, **promoted 16:31; it is what the paper now reads.**

| quantity | R3 (d01b8bae) | I2, α −0.117 | I2, α −0.216 |
|---|---|---|---|
| world: stay probability (median) | 0.107 | 0.142 | 0.141 |
| world: awake share | 1.8 % | 2.1 % | 2.1 % (0.02097; it does not depend on α) |
| world: outside entry share | 0.837 | 0.753 | 0.755 |
| K0 outside menu share (median) | 0.875 | 0.864 | 0.864 |
| step-2 fe/split κ̃ at κ 0.5 (α̂) | — | 0.497 | 0.498 (α̂ −0.2159) |
| sorting κ 0: h 0.9 / 0.7 / 0.5 | 0.105 / 0.362 / 0.762 | 0.107 / 0.376 / 0.766 | 0.107 / 0.372 / 0.770 |
| sorting κ 0.5: h 1.0 / 0.9 / 0.7 / 0.5 | 0.507 / 0.630 / 0.980 / 1.714 | 0.497* / 0.607 / 0.877 / 1.291 | 0.498* / 0.611 / 0.904 / 1.386 |
| sorting max excess | 0.521 | 0.098 | 0.193 |
| outside "in" κ 0.5 / 1 / 2 | 0.507 / 1.200 / 10.000 (box) | 0.497 / 0.947* / 2.008 (0.443) | 0.498 / 0.998 / 2.236 (0.669) |
| outside "out" κ 0.5 / 1 / 2 | −0.047 / −0.103 / −0.123 | −0.006 / −0.054 / −0.102 | −0.014 / −0.078 / −0.171 |
| RC κ̃ (se) κ 0.5 / κ 1 | 0.388 (0.328) / 0.072 (0.137) | 1.659 (51.3) / 2.359 (799.8) | **10.000 (0.401), at the κ box*** / 5.414 (149.7)* |
| RC α (fitted) | −0.135 | −0.200 | −0.306 / −0.307 |
| RC logit refit κ 0.5 / κ 1 | 0.467 / 0.875 | 0.455 / 0.822 | 0.450 / 0.805 |
| RC inflow ratio p10 / p50 / p90 | 0.22 / 0.66 / 1.44 | 0.05 / 0.62 / 1.90 | 0.05 / 0.62 / 1.88 |

\* = not converged (maxit or "no descent step"). Both RC rows are unconverged in every vintage. Step 7's DGP is E3's *extended* fit; n 83,507, 1,439 markets, 200 draws.

**R4. K2b (step 2b): every fit, R3 (α −0.117) → I2 (α −0.216).** `Rout/k2b_pix_change.csv`, promoted 16:31; truth: level 1.0, change −0.5, dose −0.15, α −0.2161.

| key | fe before-after | fe dose | ideal before-after | ideal dose |
|---|---|---|---|---|
| savings level (se) | 1.006 (0.042) → **0.879 (0.040)** | 0.934 (0.037) → **0.821 (0.040)** | 0.974 (0.012) → **0.925 (0.027)** | 0.979 (0.016) → **0.940 (0.029)** |
| savings change (se) | −0.501 (0.040) → **−0.420 (0.028)** | −0.484 (0.032) → **−0.379 (0.030)** | −0.490 (0.008) → **−0.458 (0.019)** | −0.492 (0.011) → **−0.477 (0.020)** |
| savings dose (se) | | −0.141 → **−0.133 (0.007)** | | −0.147 → **−0.158 (0.006)** |
| α̂ | −0.117 → −0.2155 | | −0.117 → −0.2161 | |
| checking level / change | 0.479 / −0.220 → 0.632 / −0.276 | **at bound (10 / 12) → 1.234 / −0.759** | 0.654 / −0.298 → 1.259 / −0.723 | 0.621 / −0.284 → 1.122 / −0.613 |
| time-deposit level / change | 0.953 / −0.486 → 0.910 / −0.452 | 0.931 / −0.516 → 0.832 / −0.396 | 1.040 / −0.508 → 0.988 / −0.494 | 1.042 / −0.504 → 0.995 / −0.500 |
| time-deposit dose | | −0.112 → −0.124 | | −0.173 → −0.154 |
| converged | yes → yes | **no → yes** | yes → yes | yes → yes |

- **Other keys:**
  - `alpha_P.max` 0.000;
  - `chk.at_bound.fe_dose` 1 → **0**; `converged.fe_dose` 0 → **1**;
  - **`gate.error` 0.001 → 0.080** (gate GATE_CHANGE 0.05; recorded `k2b_gate.ok = false`);
  - n: fe 76,088 → 83,507, ideal 193,659 → 205,891; markets: fe 1,443 → 1,439, ideal 1,683 → 1,719;
  - the own-savings variant has level and change at their box (−4 and −12), as before.
- **The ideal-world fits** ran without a code change: a scratch driver calls the script's own `k2b_world(ctx, "ideal")`, the call `step2b()` makes after the gate. `diag_k2_nls_simulation.py` is still `f517111f`. Every pooled fit converged, and its two starts agree.
- **`tab_k2b_pairs.tex` is rebuilt** (16:31; md5 a36042b9; `.md` d32ff098). The cells are the I2 column above. **Its notes' fixed sentence "The FE dose fit stopped at the iteration cap because the checking pair ran to the edge of its range, printed as at bound" is now wrong:** the fit converged and no cell prints "at bound". "Its two starts agree" still holds (0.820835 on both).

**R4b. The main thread's 21:47 checklist at α −0.216**

| claim | verdict | numbers |
|---|---|---|
| 7.3 (a) "The headline object is recovered" | **NO** | change errors 0.080 / 0.121 / 0.042 / 0.023 (2.8 / 4.0 / 2.2 / 1.2 se); dose fe −0.133 (2.6 se off), ideal −0.158 (1.4 se). Only the ideal dose fit is within about 1 se |
| (b) the savings change lies between fe_ba and fe_dose | **NO** | the range is −0.477 (ideal dose) to −0.379 (fe dose); fe_ba −0.420 is inside it |
| (c) the price coefficient is recovered | **YES** | α̂ −0.2155 (fe) and −0.2161 (ideal) against −0.2161; \|α_P\| ≤ 0.0003 |
| (d) "below the truth by more than its se in three of four fits" | **now four of four** | level gaps 3.0 / 4.4 / 2.8 / 2.1 se; the text's examples (fe dose, ideal before-after) still hold. Appendix C's "the level runs from fe_dose to fe_ba" fails: the range is 0.821 (fe dose) to 0.940 (ideal dose) |
| (e) "the dose fit … stopped at its iteration cap" | **NO** | it converged |
| (f) the checking pair is not pinned down even where the common denominator holds | **YES** | ideal 1.259 / 1.122 against 1; −0.723 / −0.613 against −0.5. Appendix C's "runs to its bounds once the dose term is added" is **NO**: fe dose is 1.234 / −0.759, not at a bound |
| (g) the time-deposit dose lies on opposite sides of the truth | **YES, narrowly** | fe −0.124 against ideal −0.154, around −0.15. "The time-deposit pair recovers the level and change" holds in the ideal world (0.988 / −0.494; 0.995 / −0.500) and less well in fe (0.832 / −0.396) |
| Appendix C, "Three results carry into the design" | its first result fails in fe | "the savings pair recovers the change and the dose in both worlds"; the rest as in (b) to (g) |
| plan 133/417, "the price coefficient is that of Section estimation:demand" | **now TRUE** | the simulation uses E3's ext1 BLP mean coefficient, −0.216 (`demand.t6.alpha.E3`) |

**Yesterday's verdicts, re-checked at α −0.216:**
- **Plan 427 (sorting): holds.** The excess over κ − ln h at κ 0.5 is 0.006 / 0.048 / 0.193, rising with sorting; "by up to" is now **0.19**.
- **Plan 429 (outside pool):**
  - "about zero" is looser at κ 2 (out −0.171);
  - κ 2 in 2.236 (0.669), converged;
  - **the plan's new "the κ 1 fit stopped at the cap" is now FALSE**: it converged (0.998).
- **VF 714: holds** (86 percent, 0.14).
- **RC rows:** "far above the truth and uninformative" holds. κ 0.5 is at the κ box, 10.000.

**Code, files and side effects:**
- `diag_k2_netflow_simulation.py` `55787e72` → `1cc8468f` (ALPHA_BAR read from `blp_results_E3_spec_12_ext1.json`, −0.21609523, guarded; the file keeps its CRLF endings).
- **`make_paper_numbers.py` → `d09d050b9015ff99c7688915cd1ab5a5`** (CR 0):
  - `K2_STEP7_DIR = "k2_fa38794d_a216"` (Kappa agent).
  - IMPR fix, 16:32: `collect_k2_rc` also accepts the step-7 `rc` record in the json, because step 2b, run later into the same folder, had rewritten `last_run`. Before the fix the RC keys were absent.
  - Rebuild rc 0, **0 MISMATCH, ok 333**.
- **Backups** of the replaced top-level files are in `Rout/k2_fa38794d_a117_toplevel/` (md5-verified):

  | file | backup md5 |
  |---|---|
  | `k2_numbers.json` | 95abf205 |
  | `k2_sorting.csv` | 64dfac1d |
  | `k2_outside.csv` | 253c75ba |
  | `k2b_pix_change.csv` | eb4588ab (d01b8bae) |
  | `k2_nls_numbers.json` | 77f418e6 (d01b8bae) |

- **Promoted:**

  | file | md5 |
  |---|---|
  | `k2_numbers.json` | 83809d03 |
  | `k2_sorting.csv` | 2d72967c |
  | `k2_outside.csv` | 12f19dd1 |
  | `k2b_pix_change.csv` | 3b952598 |
  | `k2_nls_numbers.json` | 36ee2ac2 |
  | `k2_rc.csv` (in the a216 folder) | 48853256 |

- **Side effect:** `kappa/k1_netflow_nls.py` test 1 now **fails** (max |estimate − stored| 0.104), because its stored d01b8bae recovery was simulated at α −0.117. The bootstrap check still reproduces V to 1.8e-15. Not changed.

**R5. Other results of 11-08 and 11-09** (the detail is in items 1e–1i)

| item | result |
|---|---|
| Pix roster (1e) | 16 mandated B conglomerates; B savings 2020Q4 R$ 1,021.9 bn; non-mandated R$ 2.94 mn (Bank of China); off-Pix max R$ 2.94 mn in 2020Q4 |
| Big-5 branches (1f) | 94.4 % of 3,207 MCAs (3,027); 99.1 % of population; 1 MCA without a B branch (`UNKNOWN_1504752`) |
| Firm-type fallback (1g) | B 0.917 [0.893, 0.938], identical to the main estimate; D 0.978 [0.891, 0.999] vs main 0.980; D MAE 45 vs 99 bp (B 32 vs 52); D holdout n 592 → 449 |
| State vector (1h) | D gap −0.004 → **+0.001** (sign flip, VF 466); effective dimension 2.591 / 2.413 / 2.356; condition number 7.3 → 29.9 with the young share; the table rewritten |
| E4 extended (1d) | e3ext adopted (Q 0.0169365, σ 0.384, α −0.700, SE refused K > L); local in sync |
| Script md5s (LF) | `make_paper_numbers.py` `d09d050b`; `diag_k2_netflow_simulation.py` `1cc8468f`; `diag_k2_nls_simulation.py` `f517111f` (unchanged); `kappa/k1_netflow_nls.py` `7430007a` |
