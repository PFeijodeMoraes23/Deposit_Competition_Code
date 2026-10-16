# Handoff: descriptive evidence on depositor inertia (prices, balances, relative prices, entry)

**Date:** 2026-10-13, first written 16:20, updated 18:20 (state at the end of the session).
**From:** the motivation-figures session. **To:** (a) an agent asked to *think about* this
evidence: interpret it, stress-test it, reconcile it with the model, and propose what to do next;
(b) a session that resumes the work (section 9 lists the state, the files and what is pending).

**UPDATE 2026-10-15 (read first):** Pedro removed the Figure 2 robustness analysis (evidence G,
the note's section 4.8, Figure 15, Table 10, `diagnostics/diag_entry_concentration.py` and its
outputs) and the interquartile-range version of Figure 1 (Figure 1 is now the +/-1 SD band). G is
kept below as a record only: its files no longer exist. The current state of the note, with the
construction of every exhibit, is in `claude/motivation_note_writing_handoff_2026-10-15.md`.

**Rules:** V_Main.tex is edited by the author only. The deck
(`Drafts/Presentations/Deposit Competition/2026-10-06/Presentation-2026-10-06.tex`) is open in
TeXstudio and is never edited. The data pipeline is read-only. Use
`C:/venvs/egan/Scripts/python.exe` (importing project modules from another interpreter runs them).
Do not mention or use git unless Pedro asks. Bring open choices to Pedro as questions.

**Reader-facing write-up of everything below:**
`Drafts/Deposit Competition/motivation_figures_review.pdf` (37 pages, written for professors; md
source next to it; moved there from `Drafts/Presentations/Deposit Competition` at 18:00 on
Pedro's request). Its ten tables also exist as LaTeX bodies, `tab_motiv_*.tex`, in the same
folder (section 9). Every number below is also a `\pn` key under `slide.motiv.*` in
`paper_numbers_macros.tex`; `PAPER_NUMBERS/paper_numbers.md` lists them with definitions.

## 0. Basis

```text
files[15]{file,role,saved}:
  BCB/.../processed/market_panel.parquet,all inputs (08-26 build; not rebuilt since),08-26 18:23
  Code/.../make_slide_figures.py,every chart + the sidecar + the tab_motiv_*.tex tables,10-13 18:15
  Code/.../make_paper_numbers.py,collect_slide_motivation -> slide.motiv.* keys,10-13 17:04
  Code/.../diagnostics/diag_entry_concentration.py,Figure 2 robustness (read-only),10-13 13:48
  Code/.../sleep_ident_entry_dynamics.py,Figure 2 / D6 (other session's window rule),10-12 18:25
  .../ESTIMATION_OUTPUT/Rout/slide_motivation.json,unrounded numbers of every chart and table,10-13 17:22
  .../ESTIMATION_OUTPUT/Rout/slide_anova_tables.md,Tables 4-7 as md (spliced into the note),10-13 17:22
  .../DIAG_PHI_SEPARATION/diag_entry_concentration.{csv,json},robustness results,10-13 13:56
  .../DIAG_PHI_SEPARATION/d6_entry_events.csv,entry events (window rule),10-12 18:30
  .../DIAG_PHI_SEPARATION/d6_implied_phi.csv,paper's phi_entry,10-12 18:33
  .../PAPER_NUMBERS/paper_numbers_macros.tex,slide.motiv.* keys (1,521 values; only additions),10-13 17:09
  Drafts/Deposit Competition/motivation_figures_review.{md,pdf},professor-facing note (37 pp),10-13 18:26
  Drafts/Deposit Competition/tab_motiv_*.tex,the note's 10 tables as booktabs bodies,10-13 18:16
  Code/.../claude/motivation_note_build/,build.sh + splice_anova.py + prof.lua + prof_header.tex,10-13 18:05
  Code/.../claude/entry_paths_handoff_2026-10-12.md,the Figure 2 session's handoff,10-12 19:02
```

The sidecar was re-saved at 17:22 by a rerun of `--only anova` that changed only the md layout of
Table 6; the 17:09 keys carry the same values. Re-run any chart before trusting a number that
differs from this brief.

## 1. Ground truth (what the code does)

| Element | What the code does | Where |
| --- | --- | --- |
| Sample | 2016Q1–2024Q4 (`utils/window.py`), paper sample via `mdt.prepare_panel` (no dep_a5/spread_a5 before 2020Q4) | `make_slide_figures.motiv_panel` L.643 |
| Firm types | B = branch banks, balances by municipality (ESTBAN) aggregated to MCAs; D = digital, national rows (IF.Data, `mca_code` NATIONAL) | `is_B` column, panel_6 |
| Spread | Selic minus deposit rate, compounded: 100·[(1+rf)^4 − (1+r)^4], pp | `mdt.annualized_spread_pp` |
| Type-4 rate | COSIF interest expense on time deposits / lagged stock; ONE national rate per conglomerate; fallback chain ends at CDI; above 3 × Selic → CDI (`rate_a4_guarded`) | `panel_deposit_rates.py` |
| "Own rate" | rate not equal to the quarter's CDI (1e-9) and not guarded. 11% of B bank-quarters and 61% of D (15% of D balances) fail it | `bank_quarter` L.1089 |
| B means | Table A.4 weights: bank's type-4 share of the MCA-quarter × MCA-quarter total balance (mean equals Table A.4's line to 2e-14) | `b_spread_cells` L.656 |
| Intervals | Conglomerate wild cluster bootstrap, Webb, 999 draws, bias-corrected percentile, via `utils.sleep_links.cluster_wild_bootstrap` + `_ame_band_from_draws`; time averages keep one weight per bank across quarters | `_mean_ci` L.694, `_wcb` L.720, `average_ci` L.746 |
| FE slopes | `_fe_slope` (dummies, cluster OLS + `linear_wild_cluster_bootstrap`); `_wls_slope` (weighted alternating projections, sqrt(w) OLS, same bootstrap) | L.1107, L.1598 |
| Entry events | D6 events with the 10-12 window rule (entries dated 2016Q1–2024Q4, paths cut at 2024Q4); share = entrant's deposits (types 1, 2, 4) / all B firms' deposits in the MCA, from balances only | `sleep_ident_entry_dynamics.build_events`; rebuilt in `entry_gap_chart` L.1330 (exact match to d6_entry_paths.csv) |
| Entry-implied φ | Model Dep_h = A + φ g_h Dep_{h−1}, per-entrant accrual g_h; normalization (s_h − s_0)/(s_end − s_0), s_end = mean h 10–12; grid 1,400 points on [0.50, 0.9985]; min squared distance median model vs median data over h = 1..9; bootstrap 400 draws | `implied_phi` L.492 (sed) |

## 2. Methods by analysis (one line each)

```text
analyses[9]{id,question,method,chart}:
  A,dispersion across B banks,Table A.4-weighted SD and P25-P75 of the type-4 spread by year,fig_slide_selic_timedep_all_{sd,iqr}
  B,prices vs balances,quarterly spreads/rates by type (bank means with WCB bands) and shares of balances; hiking cycle from the Selic target,fig_slide_motiv_{spreads,shares,rates}_by_type
  C1,do banks move together,sequential weighted R2: quarter FE (common) then bank FE (persistent); local gap: bank x quarter FE vs rest,fig_slide_relprice_decomp
  C2,persistence of relative price,gap at t+4 on gap at t (pp; % of CDI; local),fig_slide_relprice_persist_*
  C3,do shares follow relative prices,per-bank cross-section (avg gap vs yearly log share change); bank-quarter panel (gap_t vs log share change t->t+4; bank+quarter FE); local raw and within-bank (bank x quarter + market FE),fig_slide_relprice_share_*
  D,own repricing,quarterly change in own spread vs log time-deposit growth t->t+4; bank+quarter FE; B and D firms; trim 1/99; floor R$100m,fig_slide_motiv_reprice_binscatter{,_d}
  E,D-firm premium,B mean spread minus D mean spread with joint WCB; D share of time deposits,fig_slide_motiv_dfirm_premium
  F,entrants by price at entry,gap = entrant spread - incumbents' dep-weighted spread at entry; split >=1pp more vs rest; medians with event and BANK bootstraps,fig_slide_motiv_entry_by_gap{,_level}{,_bankboot}
  G,Figure 2 robustness,implied_phi re-run: banks resampled; without Santander; one vote per bank (weighted medians both sides); Santander only,fig_slide_phi_entry_robustness
```

## 3. Evidence

**A. Dispersion (pp a year, balance-weighted across B firms).** IQR 2.6–4.6 pp in 2017–2024, 7.2
in 2016. SD 1.8–3.4 pp (2017–2024), 7.4 in 2016 (inflated by Bradesco's HSBC consolidation
quarter, 2016Q3, spread −34 pp).

**B. Prices vs balances, hiking cycle 2020Q4 → 2022Q3 (Selic target 2.00 → 13.75).**

```text
hikes[9]{series,2020Q4,2022Q3}:
  demand spread (= Selic) pp,1.961,14.145
  savings spread pp,0.552,5.570
  time spread B pp [95% CI],-0.485 [-1.709 0.694],3.745 [2.032 5.306]
  time spread D pp [95% CI],-0.896 [-1.830 0.031],2.674 [-0.080 5.570]
  demand share %,9.170,7.962
  savings share %,32.405,27.129
  time D share of all balances %,4.805,7.873
  prepaid share %,0.427,0.975
  savings share drift pp/quarter (before / during / after),-0.507 / -0.754 / -0.595,
```

Whole window: savings share 42.0 → 21.8%, time deposits at D firms 2.0 → 10.6% of all balances.
Nominal savings balances −4.4% over the hikes. Prepaid spread ≈ 0 (|max| 0.002 pp). Mean CI
half-width of the quarterly bank means: B 1.905 pp, D 2.130 pp.

**C. Relative prices across B banks (B firms with an own rate; top 5 hold 81.3% of B time deposits).**

```text
decomposition[3]{price,common,bank_persistent,rest}:
  spread pp,0.308,0.325,0.367
  rate % of CDI,0.385,0.251,0.364
  local gap pp,national 0.910,,rival mix 0.090
relprice_slopes[9]{test,slope,ci95,n,banks}:
  persistence gap pp (t+4 on t),0.577,[0.409 0.740],1519,65
  persistence % of CDI,0.468,[0.322 0.618],1516,66
  persistence local gap,0.487,[0.335 0.638],218463,65
  per bank: avg gap pp -> yearly log share change %,-2.992,[-4.450 -1.613],60,60
  per bank: avg gap CDI points -> yearly share change,0.221,[0.127 0.314],60,60
  panel: gap pp -> share change next 4q % (bank+quarter FE),-0.990,[-1.742 -0.196],1516,65
  panel: gap CDI points -> share change,0.034,[-0.015 0.085],1519,66
  local raw: gap to rivals -> local share change,-1.106,[-1.543 -0.644],218458,66
  local within bank (bank x quarter + market FE),-0.777,[-2.206 0.716],218458,66
```

**D. Own repricing (slope = % of time deposits per pp of spread change).**

```text
reprice[6]{firms,growth_window,slope,ci95,n,banks}:
  B,next 4 quarters (main),-0.536,[-1.110 0.066],1349,63
  B,next 4 quarters untrimmed,-0.133,[-0.576 0.345],1408,63
  B,same quarter (mechanically biased),-1.251,[-1.718 -0.729],1575,65
  D,next 4 quarters (main),0.143,[-0.272 0.535],835,60
  D,next 4 quarters untrimmed,-0.004,[-0.423 0.418],868,62
  D,same quarter (mechanically biased),-0.476,[-0.783 -0.173],1052,65
```

Median within-bank SD of four-quarter growth: 27% (B), 23% (D). B sample covers 63 of the 67 B
conglomerates that hold time deposits at all.

**E. D-firm premium.** Mean B − D spread over 2016–2024: 2.503 pp [0.500, 4.443]; 15 of 36
quarterly intervals exclude zero; D share of time deposits 4.1 → 15.1% (1.2 pp a year).

**F. Entrants by price at entry (window rule: 201 kept B entries, 198 followed 12 quarters).**
Entrants paying more than incumbents: 62.2% (median gap 1.26 pp); at least 1 pp more: 53.7%.

```text
entry[4]{group,h4,h8,h12,ci_h12_bank_bootstrap}:
  normalized pays>=1pp more (107 entries 9 banks),0.505,0.808,1.044,[0.919 1.050]
  normalized rest (91 entries 14 banks),0.457,0.770,1.058,[0.966 1.097]
  level % of B deposits pays more,1.166,1.971,3.042,[0.019 3.688]
  level % of B deposits rest,1.080,1.571,2.008,[0.055 5.176]
```

With the EVENT bootstrap the level intervals at h12 were [2.286, 3.490] vs [1.351, 2.831] and
suggested a difference. The bank bootstrap erases it. Six banks sit in both groups.

**G. Figure 2 robustness (paper's own estimator, reproduced exactly first).**

```text
phi_entry[6]{variant,estimate,ci95,entries,banks}:
  all entries / events resampled (paper),0.914,[0.891 0.937],198,17
  all entries / banks resampled,0.914,[0.882 0.9985],198,17
  without Santander / events,0.980,[0.928 0.9985],78,16
  without Santander / banks,0.980,[0.882 0.9985],78,16
  one vote per bank / banks,0.944,[0.785 0.9985],198,17
  Santander only / events,0.878,[0.859 0.908],120,1
```

0.9985 is the top of the grid, so those upper bounds are censored. The deposit-persistence
estimates are 0.983 (III) and 0.981 (IV). The 201 kept entries come from 17 conglomerates. Santander
(C0080185) makes 122 of them, spread over 2018Q1–2022Q4 (peak 17 in 2020Q3), all with branch
records, none with deposits before the branch. The other 79 cluster in 2016. D entries: φ 0.986
[0.888, top of grid], 59 events (paper).

**H. Added 17:05 (Pedro's second review): decompositions for D firms and by period, and four
variance and co-movement tables** (`make_slide_figures.py --only relprice,anova`; sidecar
sections `relprice.decomposition_d`, `relprice.decomposition_by_period`, `anova`; tables written to
`Rout/slide_anova_tables.md`).

```text
decomp_more[8]{sample,common,bank_persistent,rest,n,banks}:
  D firms spread pp,0.232,0.282,0.487,1526,85
  B 2016-2018,0.148,0.391,0.461,654,59
  B 2019-2021,0.305,0.327,0.368,580,60
  B 2022-2024,0.186,0.536,0.277,691,62
  D 2016-2018,0.061,0.463,0.476,364,45
  D 2019-2021,0.082,0.188,0.731,411,58
  D 2022-2024,0.190,0.361,0.448,751,79
  (sequential: quarter FE then bank FE; balance-weighted),,,,,
levels_shapley[6]{firms_var,common,bank,rest}:
  B log share of time deposits,0.002,0.965,0.033
  B log time deposits,0.071,0.899,0.031
  B quarterly growth,0.111,0.057,0.831
  D log share,0.008,0.925,0.067
  B spread pp (Shapley),0.322,0.311,0.367
  D spread pp (Shapley),0.225,0.289,0.487
comove_by_horizon[4]{firms,h1,h4,h8,h12}:
  B slope % per pp,-0.305***,0.345,1.453*,2.301***
  D slope,-0.098,0.452,1.606*,6.115***
  (change in gap vs change in log share over k quarters; quarter FE; h1 mechanically negative),,,,
  B correlation (balance-weighted since 10-15),-0.109,0.051,0.146,0.188
covdecomp[2]{firms,cov,between_share,corr,corr_between,corr_within}:
  B,1.109,0.881,0.284,0.375,0.249
  D,0.760,0.733,0.124,0.159,0.134
local_nested[3]{var,market,bank,bank_in_market,rest}:
  log local share,0.252,0.512,0.162,0.075
  log local time deposits,0.855,0.082,0.030,0.033
  gap to local rivals,0.009,0.432,0.010,0.550
```

Readings to stress-test: (i) prices are ~2/3 relative while shares are ~96% fixed bank effects
(the cleanest inertia statement); (ii) LEVELS of relative price predict slow share losses
(negative), but CHANGES co-move POSITIVELY at long horizons (banks raise spreads as their base
grows: harvesting, or the stock rate lagging for banks adding cheap new deposits); (iii) high-spread
banks hold large shares (between-bank covariance 88%). Also: prepaid spreads are −0.006 bp for B and
D with zero-width intervals (CDI fallback), so prepaid prices carry no information; the five-bank
paths now use a validated categorical palette with per-rank dashes.

## 4. Conclusions as I read them (with confidence)

```text
conclusions[7]{claim,confidence,main_threat}:
  relative prices vary a lot and mostly at the national level,high,stock-rate measurement inflates idiosyncratic variance
  balances drift slowly against large price swings (savings share drift ~0.5-0.75 pp/q),high,drift predates the hikes; after-tax gap not computed
  shares follow relative prices slowly (~1%/yr per pp in panel; ~3%/yr per pp cross-section),medium,reverse causality (banks that want to grow raise rates) and noise attenuation pull in opposite directions
  own quarter-to-quarter repricing does not detectably move deposits (B or D),medium,imputed stock rate cannot date posted repricing; noise attenuates
  entrants that pay more do not accumulate faster (normalized),medium,only 17 banks; the split is close to a split of banks
  the level difference for entrants paying more is NOT established,high,
  Figure 2's phi_entry 0.914 is Santander-driven; without Santander it is 0.980 and agrees with the persistence estimates,medium-high,17 clusters with one at 61% weight make every bootstrap shaky
```

## 5. Honesty flags (keep visible)

- The time-deposit "price" is an imputed cost on the stock, not a posted rate. It lags Selic when
  Selic falls (B-mean rate ≈ 190% of CDI in 2020), and gaps include book composition. As a % of
  CDI it is especially noisy.
- Same-quarter growth regressions are mechanically biased toward a response (expense over the
  lagged stock). Only the forward (t → t+4) versions are the evidence.
- D firms: 61% of quarters at the CDI fallback, so D prices are weak, and the D premium is
  probably understated.
- Every price-quantity relation here is a correlation. Banks set rates, expansion and marketing
  jointly.
- Entry evidence: 17 banks, one at 61% of events. Bootstraps with 17 clusters are themselves
  imprecise, so read the intervals as indicative.
- Prepaid spreads are real but print as 0.000–0.002 at three decimals. Units were not changed
  (Pedro's call).
- Pre-2016 ESTBAN balances are doubled (double glob, known since 10-05). This is irrelevant
  under the window rule, but never use 2015 balances.

## 6. Questions for you to think about

1. **Do the reduced-form slopes match the model?** Hypothesis, not checked, so verify the units.
   The deck reports a price coefficient of −0.43 per pp (III), an active own-price elasticity of
   −1.61, and a mean sleeping share of 0.983. That gives an active semi-elasticity of about
   −0.43 per pp and an awake share of 1.7% a quarter. The implied stock semi-elasticity is then
   about −0.0073 per pp per quarter, or about −2.9% over four quarters for a persistent 1 pp gap.
   Observed: panel −0.99% (but the gap is only ~58% persistent), cross-section −3.0% a year. Is
   this a coincidence, or a model-consistent moment worth putting in the paper?
2. **Is the relative-price variation mostly measurement?** One test: does the idiosyncratic part
   load on Selic changes (the signature of a lagging stock)? Another: compare the cross-bank
   ranking with the Open Finance posted-CDB snapshot. It covers 33 institutions on 2026-06-30, in
   `Code/Open_Fin/Open_Finance_Debt/raw/OF_PRODUCTS/opendata-investments_bank-fixed-incomes/`,
   outside the sample window, so it is only a level check.
3. **Why do Santander's new locations fill faster** (φ 0.878 vs 0.980)? Candidates: conversion
   of payroll or corporate relationships, branch formats, a 2018–2022 interior expansion. Which
   φ should the paper report? Pooled, one vote per bank, or excluding Santander with a stated
   reason?
4. **Inference with 17 clusters, one dominant.** Is a cluster bootstrap credible here? Consider a
   leave-one-bank-out jackknife, randomization inference, or reporting by bank.
5. **Framing for the deck.** "Deposits move slowly, not with price" (own repricing n.s.) and
   "deposits follow relative prices at ~1%/yr per pp" (panel and cross-section significant): how
   can both be true and presented together? Horizon and persistence of the gap seem to be the
   bridge.
6. **The savings → time drift.** It predates the hikes and sped up from 0.51 to 0.75 pp a
   quarter. Can the acceleration be read quantitatively as the awake share responding to a
   ~5 pp savings spread? The after-tax gap is the missing input.
7. **Power of the within-bank local test.** It is n.s. with only 9% of local variation. What
   minimum detectable effect does it have, and is it informative at all?

## 7. Open decisions for Pedro (not yours)

```text
decisions[7]{item,status}:
  framing of the motivation slides (relative prices at ~1%/yr per pp vs not with price; now also the positive 12-quarter co-movement),open
  paper's Figure 2: bank-level interval and/or without-Santander robustness,open
  Figure 2 caption L.755 (other session's T1 replacement text),open; user pastes
  V_Main L.739 s_end = average over quarters 10-12 (code: observed quarters only; 14 of 198 stop before q12),user edits himself (10-15)
  deck speaker-note comments with stale typed numbers (L.556 557 992 993),optional; user pastes
  which slides use which figure or table; Slide 3 reword,open
  prepaid spreads,DECIDED 10-13: basis points in Table 2
  Figure 4b (no shading) and Figures 5b/5c,DECIDED 10-13: main body right after Figures 4 and 5
```

## 8. Reproduce

```text
commands[5]{what,command}:
  every motivation chart + sidecar + tables,C:/venvs/egan/Scripts/python.exe make_slide_figures.py --only band,pricevol,reprice,dpremium,entrygap,relprice,anova,motivtables --png
  keys (back up paper_numbers_macros.tex and diff after),C:/venvs/egan/Scripts/python.exe make_paper_numbers.py
  Tables 4-7 into the note's md,C:/venvs/egan/Scripts/python.exe claude/motivation_note_build/splice_anova.py
  the note's PDF (Git Bash; stops on any overfull line),bash claude/motivation_note_build/build.sh
  tables only (no re-estimation; reads the sidecar),C:/venvs/egan/Scripts/python.exe make_slide_figures.py --only motivtables
```

The md's other tables (1-3, 8-11) and every number in its prose are TYPED: after any rerun,
compare them with the sidecar or with the `tab_motiv_*.tex` files, which are generated.

Each chart run prints its checks: band mean = Table A.4 (2e-14), B and prepaid means = Table A.4
annual lines, entry paths = d6_entry_paths.csv (2e-16), bank-bootstrap medians = event medians,
diagnostic reproduces d6_implied_phi.csv exactly. A failed check stops the run.

## 9. State at the end of the session (18:20) and what is pending

**Done (Pedro's three review rounds on the note, all applied):** Figures 2 and 4 lost the grey
"95% interval" legend item (bands explained in captions); Figure 4b = Figure 4 without the hike
shading; Table 2 in two panels (spreads with intervals, prepaid in bp; shares) with notes on
weights and on why demand and savings spreads have no interval; Figures 5b (D firms) and 5c (by
period); longer captions for Figures 6-11, 13 and A5-A11; Figure 14 caption and a subsection on
why it differs from A13; A2-A4 on a validated five-colour palette with per-rank dashes; an
appendix intro on main versus appendix figures; an "Open questions" section; the four variance
tables (4-7) in section 4.4.

**Tables as LaTeX (Pedro, 18:00: "a .tex-ready file that we can just call using input").** One
booktabs body per table of the note, written by `make_slide_figures.py --only motivtables`
(`motivation_tables`) to `Drafts/Deposit Competition`, the same convention as
`tab_slide_deposit_types.tex`: no float, no caption, `\input` inside your own `table`. Needs
`booktabs` only. Captions and notes stay in the note's md.

```text
tables[10]{note_table,file,sidecar_section}:
  1 spread dispersion by year,tab_motiv_spread_dispersion.tex,band.by_year
  2 tightening (panels A spreads incl. prepaid bp / B shares),tab_motiv_tightening.tex,pricevol.hikes
  3 relative prices and share changes,tab_motiv_relprice_share.tex,relprice.share
  4 variance of prices and volumes (Shapley),tab_motiv_anova_levels.tex,anova.levels
  5 co-movement by horizon,tab_motiv_anova_comove.tex,anova.comove
  6 covariance between/within banks,tab_motiv_anova_covdecomp.tex,anova.covdecomp
  7 local nested variance (B),tab_motiv_anova_local.tex,anova.local
  8 own repricing B and D,tab_motiv_reprice.tex,reprice + reprice_d
  9 entrants by price at entry (bank-bootstrap interval),tab_motiv_entry_by_gap.tex,entrygap
  10 (appendix) repricing sample construction,tab_motiv_reprice_samples.tex,reprice.waterfall + reprice_d.waterfall
```

**Pending for Pedro (not for an agent):** the decisions in section 7. (The deprecated internal
notes, `motivation_figures_internal_notes.{md,pdf}`, are deleted.)

**Not done, offered only:** the note's PDF still sets its tables from the md pipe tables, not
from the `.tex` files (the md stays readable in a preview that way). Switching the PDF to
`\input` them through pandoc raw-LaTeX blocks is possible if Pedro wants one source.
