# Handoff to the writing agent: the motivation note (depositor inertia evidence)

**Date:** 2026-10-15. **From:** the session that built the note's evidence (10-12 to 10-15).
**To:** an agent specialized in writing, revising the prose of the note. **Basis:** the note and its
exhibits as rebuilt on 10-15 (section 9 lists files and times).

**Rules.**

- `V_Main.tex` (the paper) and the slide deck are edited by Pedro only. Do not edit them. If the
  note's material should go into the paper or the deck, write the proposed text in your reply.
- Edit only `Drafts/Deposit Competition/motivation_figures_review.md`, the note itself. Never
  change a number by hand. Every number traces to a generated file (section 5).
- Bring open choices to Pedro as questions. Do not decide framing on his behalf (section 8).

**Data vintage (10-16):** all numbers here and in the note are on the panel rebuilt by the main
thread on 10-15 at 22:35 (exhibits regenerated 10-16 00:16). Earlier drafts of this handoff
quoted the previous panel.

## 1. The document

`Drafts/Deposit Competition/motivation_figures_review.md`, rendered to `motivation_figures_review.pdf`
in the same folder. Title: *Depositor Inertia in Brazilian Retail Banking: Descriptive evidence on
prices, balances, and entry.* It is written for professors: Pedro's advisers, reading the
motivation of the paper *Attention, Not Switching: Depositor Inertia and Deposit Competition in
Brazilian Retail Banking*. It answers three questions:

- Do deposit prices move while balances stay put?
- Do banks' prices move together, or do relative prices change?
- When a bank offers a better price, by entering a market or by repricing, do deposits follow?

Structure:

- Summary (five findings);
- Data and definitions;
- Methods;
- Results 4.1–4.7;
- Limitations;
- Open questions (seven);
- Appendix (A1–A12 and one sample table).

Each exhibit in Results has a "*Construction.*" paragraph with its equations: sample, variables,
weights, fixed effects, estimator and inference. A subagent added these on 10-15 by reading the
code. They are the authoritative description of how each exhibit was computed. Keep them
accurate when you edit them.

**Changes on 10-15 (Pedro's decisions):**

- The section on the entry-implied sleeping share was removed: the old 4.8, its figure (old
  Figure 15), its table (old Table 10), the summary point and the open question about Santander.
  Do not reintroduce it.
- Figure 1 now shows a band of plus or minus one standard deviation. The interquartile-range
  version and Table 1's quartile columns were removed.
- The appendix was renumbered: the old A2–A13 are now A1–A12.

## 2. Exhibit map (how each one is computed)

All charts and tables come from `Code/Egan_et_al_2025_Rep/make_slide_figures.py`. The function
named below produces each one. Shared definitions follow the map.

```text
figures[29]{note_label,file_stem,function,what_it_shows}:
  Figure 1,fig_slide_selic_timedep_all_sd,band_charts,annual Selic + B rate + B spread; band = +/-1 balance-weighted SD across B cells
  Figure 2,fig_slide_motiv_spreads_by_type,pricevol_charts,quarterly spread by deposit type; B and D time-deposit bank means with 95% bands; prepaid
  Figure 3,fig_slide_motiv_shares_by_type,pricevol_charts,each type's share of total balances (ratio of totals; no interval)
  Figure 4,fig_slide_motiv_rates_by_type,pricevol_charts,rates = Selic minus the Figure 2 spreads; hike shading
  Figure 4b,fig_slide_motiv_rates_by_type_noshade,pricevol_charts,Figure 4 without the hike shading
  Figure 5,fig_slide_relprice_decomp,relprice_charts,sequential variance split of B relative prices: quarter FE / bank FE / rest; local gap: national vs rival mix
  Figure 5b,fig_slide_relprice_decomp_d,relprice_charts,the same split for D firms
  Figure 5c,fig_slide_relprice_decomp_time,relprice_charts,the split by period 2016-18 / 2019-21 / 2022-24 for B and D
  Figure 6,fig_slide_relprice_persist_pp,relprice_charts,gap at t+4 against gap at t (B; binned + all points; WLS slope)
  Figure 7,fig_slide_relprice_share_xs_pp,relprice_charts,per-bank cross-section: average gap vs yearly log-share change
  Figure 8,fig_slide_relprice_share_panel_pp,relprice_charts,bank-quarter panel: gap at t vs log-share change t to t+4; bank + quarter FE
  Figure 9,fig_slide_relprice_share_local_fe,relprice_charts,local: gap to rivals vs local share change; bank-by-quarter + market FE
  Figure 10,fig_slide_motiv_reprice_binscatter,reprice_binscatter (_reprice_one),own spread change t-1 to t vs deposit growth t to t+4; B; bank + quarter FE
  Figure 11,fig_slide_motiv_reprice_binscatter_d,reprice_binscatter (_reprice_one),the same for D firms
  Figure 12,fig_slide_motiv_dfirm_premium,dpremium_chart,B minus D spread with 95% band; D share of time deposits
  Figure 13,fig_slide_motiv_entry_by_gap_bankboot,entry_gap_chart,median normalized share path of B entrants by price at entry; banks resampled
  Figure 14,fig_slide_motiv_entry_by_gap_level_bankboot,entry_gap_chart,the same in levels (% of B deposits in the market)
  Figure A1,fig_slide_relprice_paths_pp,relprice_charts,spread gap paths of the five largest B banks (size rank only)
  Figure A2,fig_slide_relprice_paths_pcdi,relprice_charts,rate as % of the CDI for the five largest
  Figure A3,fig_slide_relprice_paths_local,relprice_charts,gap to local rivals of the five largest
  Figure A4,fig_slide_relprice_persist_pcdi,relprice_charts,Figure 6 with % of the CDI
  Figure A5,fig_slide_relprice_persist_local,relprice_charts,Figure 6 with the local gap (binned only)
  Figure A6,fig_slide_relprice_share_xs_pcdi,relprice_charts,Figure 7 with % of the CDI
  Figure A7,fig_slide_relprice_share_panel_pcdi,relprice_charts,Figure 8 with % of the CDI
  Figure A8,fig_slide_relprice_share_local_raw,relprice_charts,Figure 9 without fixed effects
  Figure A9,fig_slide_motiv_reprice_binscatter_points,reprice_binscatter,Figure 10 with every bank-quarter drawn
  Figure A10,fig_slide_motiv_reprice_binscatter_points_d,reprice_binscatter,Figure 11 with every firm-quarter drawn
  Figure A11,fig_slide_motiv_entry_by_gap,entry_gap_chart,Figure 13 with entry events resampled (as the paper's Figure 2)
  Figure A12,fig_slide_motiv_entry_by_gap_level,entry_gap_chart,Figure 14 with entry events resampled
tables[10]{note_table,tex_file,function,content}:
  Table 1,tab_motiv_spread_dispersion,band_charts -> motivation_tables,mean and SD of the B time-deposit spread by year
  Table 2,tab_motiv_tightening,pricevol_charts -> motivation_tables,spreads (Panel A; prepaid in bp) and shares (Panel B) at 2020Q4 and 2022Q3
  Table 3,tab_motiv_relprice_share,relprice_charts -> motivation_tables,six slopes of share change on relative price (Figures 7-9 and A6-A8)
  Table 4,tab_motiv_anova_levels,anova_tables -> motivation_tables,Shapley split of prices and volumes: common / persistent / rest
  Table 5,tab_motiv_anova_comove,anova_tables -> motivation_tables,co-movement of changes in gap and log share at 1/4/8/12 quarters
  Table 6,tab_motiv_anova_covdecomp,anova_tables -> motivation_tables,covariance of gap and log share between vs within banks
  Table 7,tab_motiv_anova_local,anova_tables -> motivation_tables,nested local split: market / bank / bank-in-market / rest
  Table 8,tab_motiv_reprice,reprice_binscatter -> motivation_tables,own-repricing slopes B and D (main / untrimmed / same quarter)
  Table 9,tab_motiv_entry_by_gap,entry_gap_chart -> motivation_tables,entrants' medians at quarters 4/8/12 + bank-bootstrap interval
  Appendix table,tab_motiv_reprice_samples,reprice_binscatter -> motivation_tables,the filters that build the repricing samples
```

The `.tex` files are booktabs table bodies in `Drafts/Deposit Competition`, ready for `\input`.
The md shows the same tables as pipe tables.

**Shared definitions (the Methods section states them; keep them consistent):**

- **Sample.** 2016Q1–2024Q4, quarterly, cooperatives excluded.
- **Firm types.** B firms are brick-and-mortar banks, observed by municipality (ESTBAN) and
  aggregated to minimum comparable areas (MCAs). D firms are digital banks, observed nationally
  (IF.Data).
- **Spread.** Selic minus the deposit rate, both compounded to annual rates, in pp.
- **Time-deposit rate.** Interest expense over the lagged stock (COSIF), one national rate per
  conglomerate. It falls back to the CDI when there is no own rate or when the rate exceeds three
  times Selic. This happens in 11% of B and 60% of D bank-quarters. Analyses of relative prices
  use only quarters with an own measured rate.
- **Weights.** Means are balance-weighted. The B-firm weights reproduce the paper's Table A.4
  exactly.
- **Intervals.** 95%, from a wild cluster bootstrap by conglomerate (Webb weights, 999 draws,
  bias-corrected percentile). The per-bank cross-section uses each bank as its own cluster. The
  entry bands are percentile intervals of the median, 999 draws, with banks resampled in the main
  figures and entry events resampled in the appendix.

## 3. The argument and how strong each piece is

```text
claims[8]{claim,evidence,strength,main_caveat}:
  prices differ across banks,Fig 1 + Table 1 (SD 1.8-3.4 pp 2017-2024),high,imputed stock rate adds book-composition noise
  prices moved a lot while balances drifted slowly,Figs 2-4 + Table 2,high,savings drift predates the hikes; after-tax gap not computed
  relative prices vary but shares are fixed,Fig 5-5c + Table 4 (spread ~1/3 common; B share 96.5% bank-fixed),high,
  relative position persists about half a year-on-year,Fig 6 (0.573),medium-high,noise attenuates the slope
  deposits follow relative prices slowly,Figs 7-9 + Table 3 (-1.1%/yr panel; -2.5%/yr cross-section; local within-bank n.s.),medium,correlations; reverse causality vs attenuation
  own repricing does not detectably move deposits,Figs 10-11 + Table 8,medium,stock rate cannot date a posted repricing
  long-horizon co-movement is positive (harvesting or measurement),Table 5 + Table 6,medium,the two readings are not separable here
  entrants that pay more do not grow faster,Figs 13-14 + Table 9,medium,17 banks; one bank makes 122 of 201 entries
```

## 4. What the writing must keep visible (honesty flags)

- The time-deposit "price" is an imputed cost on the stock, not a posted rate. Do not call it an
  offered or posted rate.
- Every price-quantity relation is a correlation. Never write "causes", "drives" or "responds
  to" for these slopes. Keep the note's two biases: banks that want to grow compete in other
  ways (toward a response), and measurement noise (toward zero).
- Same-quarter slopes are mechanically biased, because expense is divided by the lagged stock.
  Only the forward (t to t+4) slopes are evidence.
- Prepaid spreads (−0.006 bp) carry no pricing information. Nearly all prepaid balances are
  priced at the CDI fallback.
- D-firm prices are weak (60% CDI fallback), so the D premium (2.701 pp) is probably
  understated.
- Entry evidence rests on 17 banks, and Santander makes 122 of the 201 entries. That is why
  Figure 14's bank-resampled bands are wide. These sentences stay. Only the separate sleeping-share
  robustness section was removed.
- The positive 12-quarter co-movement (Table 5) has two readings that the data cannot separate.
  Present both.

## 5. Where numbers come from (never retype from memory)

- The unrounded values are in
  `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/Rout/slide_motivation.json`. The
  `tab_motiv_*.tex` files and the `\pn{slide.motiv.*}` keys (in `paper_numbers_macros.tex`) are
  generated from it.
- The md's prose numbers and its pipe tables 1–3 and 8–9 are TYPED. Tables 4–7 are copied in by
  `claude/motivation_note_build/splice_anova.py`. If you change a number, check it against the
  `.tex` table or the sidecar.
- For any text that goes into the paper or the deck, quote numbers with `\pn{slide.motiv.<key>}`.
  The key families are `band`, `hikes`, `prepaid`, `dpremium`, `entrygap`, `relprice`, `reprice`,
  `reprice_d` and `anova`. `PAPER_NUMBERS/paper_numbers.md` defines each key.
- Display rules: three decimals in tables and counts as integers. Spell out "percentage points
  (pp)" and "basis points (bp)" at first use. Annualization is always compounded.

## 6. Format rules of the note (the PDF build breaks otherwise)

- Build from Git Bash with
  `bash "Code/Egan_et_al_2025_Rep/claude/motivation_note_build/build.sh"`. It writes the PDF next to
  the md and stops on any overfull line or LaTeX error.
- **Figure captions** are the alt text of `![Figure N. ...](<file.png>)`. Keep "Figure N." at the
  start (the filter sets it in bold). Put no math and no LaTeX in captions, because the filter
  escapes them.
- **Table captions** are `Table: ...` lines. Do not change the first 60 characters of the captions
  of Tables 4–7, and do not edit their rows: the splice script matches them.
- **Headings** start at `##` (sections) and `###` (subsections). The YAML `title:` is the
  document's only level-1 heading, so markdownlint (rule MD025) stays clean, and `build.sh`
  shifts every heading up one level for the PDF. Never add a `#` heading.
- Math goes in body text only, as `$...$` and `$$...$$`. There is no raw LaTeX outside math.
  Write a literal currency sign as `R\$`.
- Prose style: American English, no semicolons, no dashes separating ideas (en dashes in ranges
  are fine), acronyms defined at first use, plain short sentences, no hype.

## 7. Corrections made on 10-15 (text now matches the code; keep them)

The math subagent compared the prose with the code and found eight disagreements. I checked
each one against the code and fixed the text. No number changed.

```text
fixes[8]{where,old_text_said,code_does_and_text_now_says}:
  Methods Intervals,all intervals are bias-corrected WCB,entry bands (Figs 13 14 A11 A12; Table 9) are plain percentile intervals of the median; banks (or events in the appendix) resampled
  Section 4.4 intro + Table 5,all four decompositions weighted on the same bank-quarters,samples differ by table; Table 5's correlation was unweighted -> made balance-weighted in the code (Pedro 10-15); only that column changed (B h12 0.151; D h12 0.317 on the 10-15 panel) and each correlation now has its slope's sign
  Methods Entry events + 4.7 + Fig 13 caption,entries 2016Q1-2024Q4 followed 12 quarters,kept entries date 2016Q1-2022Q4 (8-quarter screen); 198 observed through q10; 190 through q11; 184 through q12
  Why Fig 14 differs,Figure 14 resamples the 17 banks,banks resampled separately within each group (9 and 14 banks)
  Fig 8 and Fig 9 captions,own measured rate,own rate (Fig 9: defined local gap) at both t and t+4
  Methods Weights + Binned scatter + Fig 10 caption,means balance-weighted unless stated,repricing slopes and bins (Figs 10 11 A9 A10; Table 8) are unweighted
  Fig 7 and A6 captions,(60 banks),60 banks after the 1/99 trim
  Methods entry screens,initial share below 10%; reporting gaps,share may not exceed 10%; one missing quarter allowed; entrant must have a branch record in the market; s_end averages the observed quarters among 10-12
```

## 8. Open questions for Pedro (not for the writing agent to settle)

1. **Table 5's correlation column: RESOLVED 10-15.** Pedro chose a balance-weighted correlation.
   The code, the table, the note's text and the PDF are updated. The slopes are unchanged.
2. **`V_Main.tex` line 739** says $s_{\mathrm{end}}$ is "the average share over quarters 10 to 12
   after entry". The code averages only the observed quarters among 10 to 12. 14 of 198 entries
   are not observed through quarter 12. **Pedro will make this edit himself (10-15). Do not
   propose text for it.** The same holds for the Figure 2 caption at line 755.
3. **Framing of the motivation.** Two framings are on the table: "deposits move slowly, and not
   with price", or "deposits follow relative prices at about 1% a year per pp". The positive
   long-horizon co-movement also bears on it.
4. **Use in the paper and the deck.** Which exhibits go where. Nothing from the note is in the deck
   or in `V_Main.tex` yet.
5. **Open question 1's model numbers.** It quotes the paper's demand estimates (1.7% re-optimized
   a quarter, semi-elasticity about 0.43). They change if the sleep or demand stage is re-estimated
   (a rerun is pending Pedro's decision). Check them before any circulation.

## 9. Files (basis)

```text
files[6]{file,role,saved}:
  Drafts/Deposit Competition/motivation_figures_review.md,the note (edit this),10-15 ~14:00 (math + corrections)
  Drafts/Deposit Competition/motivation_figures_review.pdf,rendered note (40 pages; no overfull lines),10-15 14:06
  Drafts/Deposit Competition/fig_slide_*.{pdf,png} and tab_motiv_*.tex,exhibits,Figure 1 redrawn 10-15; other figures 10-12/10-13; tables 10-15 13:44
  BCB/.../ESTIMATION_OUTPUT/Rout/slide_motivation.json,unrounded values,10-15 (band section rewritten; same values)
  Code/Egan_et_al_2025_Rep/make_slide_figures.py,every exhibit,10-15
  Code/Egan_et_al_2025_Rep/claude/motivation_note_build/,build.sh + splice_anova.py + prof.lua + prof_header.tex,10-13
```

The inputs have not changed since the exhibits were built: `market_panel.parquet` dates from 08-26,
the entry events from 10-12 and Table A.4's CSV from 09-28. No figure needs regenerating for the
writing work. The macro keys were last published on 10-13. The next `make_paper_numbers.py` run
drops the band's p25, p75 and IQR keys, which no document uses.
