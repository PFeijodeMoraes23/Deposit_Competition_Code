# Handoff: Figure 2 entry paths (window fix, raw-ESTBAN check, accrual g, pipeline findings)

**Date:** 2026-10-12, 19:00. **From:** the entry-paths session. **To:** the agent consolidating the bigger picture.

**Scope:** Figure 2 of V_Main (`fig:phi_entry_dyn_fixed_phi`, page 38): the median share paths of entrants and the entry-implied sleeping share (`phi.entry.*`). This brief covers the sample rule changed today, an independent raw-ESTBAN check, how the accrual factor g is built, and two data-pipeline findings.

**Rule:** V_Main.tex is edited by the author only. The deck is open in TeXstudio: never edit it.

**Basis (all local, all today unless noted):**

```
files[11]{file,role,saved}:
  sleep_ident_entry_dynamics.py,Figure 2 generator (edited today),18:26
  make_slide_figures.py,deck visuals; entry_gap_chart edited (file was already modified, staged, before this session),18:34
  diagnostics/diag_entry_dynamics_estban.py,NEW read-only raw-ESTBAN check,18:30
  diagnostics/report_entry_paths_estban.py,NEW generator of the research note,18:55
  ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/d6_*,Figure 2 outputs (window rule),18:33
  ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/_D6_PREWINDOW_20261012/,archive of the previous d6_* + paper panels + macros + slide chart,18:27
  ESTIMATION_OUTPUT/DIAG_ENTRY_ESTBAN/*_window*,raw-ESTBAN check outputs,18:47
  Drafts/Deposit Competition/paper_numbers_macros.tex,refreshed by make_paper_numbers.py,18:39
  Drafts/Deposit Competition/fig_entry_dynamics_{a,b}.{pdf,png,tex},Figure 2 panels,18:33
  Drafts/Deposit Competition/fig_slide_motiv_entry_by_gap{,_level}.pdf + Rout/slide_motivation.json,deck chart + its numbers,18:35
  Drafts/Deposit Competition/entry_paths_estban_report.{tex,pdf},research note (9 pages),18:55
```

Inputs not changed: `market_panel.parquet` (2026-08-26), `estban_own_branches.csv` (2026-08-26), `DEMAND_PREP/demand_2_spec_12.parquet`, each routine's `national_phi_t.csv` (E3, E4).

## 1. Ground truth (what the code does now)

| Element | What the code does | Where |
|---|---|---|
| Sample rule | Entries dated inside `utils/window.py` (2016Q1 to 2024Q4); quarters after 2024Q4 cut; quarters before 2016 used only to show the entrant was absent before. ON by default; `--no-window-only` restores every panel quarter | `sleep_ident_entry_dynamics.py`: `WINDOW_Q0/Q1`, `load_event_frames()`, argparse `--window-only` (BooleanOptionalAction) |
| B-firm share | Entrant's deposits (types 1, 2, 4) over all locally booked deposits in the MCA. Every local panel row is ESTBAN and every D row sits at CODMUN 0, so this is the share conditional on B firms | `load_event_frames()`, `build_events()` |
| D-firm share | Entrant's deposits over all D firms, one national market | same |
| Branch screen | Flags deposits two or more quarters before the first branch, judged only where the branch data cover the entry quarter (`branch.attrs["q_cover0"]`, which is 2016Q1) | `load_branch_entry()`, `build_events()` |
| Accrual g | Per entrant and quarter: deposit-weighted mean of `gross_return_lag` over types 1, 2, 4 of its own (congl, MCA, quarter) cells; falls back to the conglomerate's cross-market mean, then the national median, then a flat carry | `build_g_paths()` |
| `gross_return_lag` | `1 + risk_free_qoq_lag - spread_qoq_lag`, lagged within (congl, type, MCA). Type 1 rate 0; type 2 = TR + 0.5%/month (Selic target > 8.5) or TR + 70% of the target; type 4 = conglomerate COSIF implicit rate, replaced by CDI above `RATE_A4_MAX_MULT` = 3 x Selic. NOT estimated | `sleep_demand_prep_e1.py` L.219-224 (same in `_e2`, `_link`); rates in `panel_deposit_rates.py` L.1164-1174, L.102 |
| Implied phi | Grid 1,400 points on [0.50, 0.9985]; minimize the squared distance between the median model path and the median data path over h = 1..9 (plateau normalization); bootstrap of 400 entry resamples (the 999 setting is capped at 400); per-entrant g paths | `implied_phi()`, `model_tensor()` |
| Model curves | Mean of `phi_t_IV_HausmanFull_x_Tech` over 35 quarters (E3 0.983, E4 0.981), evaluated on each entrant's own g path | `phi_vintages()`, `vintage_curve()` |
| Meta | `d6_meta.json` now records `window_q` = [8064, 8099] | `_write_meta()` |
| Deck chart | `entry_gap_chart` cuts at `meta["window_q"]`, so its 1e-12 fidelity check against `d6_entry_paths.csv` passes (2.2e-16) | `make_slide_figures.py` |

## 2. Results to carry

```
results[6]{object,value,ci95,n,source}:
  phi_entry B (paper; new rule),0.914,[0.891 0.937],198,d6_implied_phi.csv
  phi_entry B (previous rule),0.917,[0.894 0.941],202,_D6_PREWINDOW_20261012/d6_implied_phi.csv
  same entries at fixed g=1.011 (control),0.915,[0.891 0.938],198,DIAG_ENTRY_ESTBAN/estban_implied_phi.csv panel_fixed_g
  raw ESTBAN institution level (same rule; fixed g),0.921,[0.899 0.942],211,estban_implied_phi.csv window
  phi_entry D (paper),0.986,[0.888 top of grid],59 main / 60 alt,d6_implied_phi.csv
  phi*g B,0.926,,198,d6_implied_phi.csv
```

- Kept B entries: 201 by cohort (2016:36, 2017:15, 2018:23, 2019:45, 2020:36, 2021:29, 2022:17). Of these, 198 are plotted; 3 entrants from 2022Q3 to Q4 are not observed through h = 10 to 12. D: kept 63, 3 censored (2022Q4), 1 dropped by the |den| < 1e-6 guard.
- The branch screen now drops 1 entry. The previous rule dropped 125, of which 124 were entries from 2014Q1 to 2015Q3 with no branch data at all.
- The raw-ESTBAN medians stay within 0.015 of the paper's at every horizon, in both normalizations.
- `phi-in-CI` (deposit-weighted fitted phi inside the B interval): E3 4.0%, E4 3.8%, E1 21.5%, E2 22.7% (`d6_phi_in_interval.csv`).

## 3. Tasks

```
tasks[8]{id,agent,deliverable,status}:
  T1,Pedro (author),V_Main caption L.755 (find/replace text under T1 below),pending
  T2,Pedro,recompile V_Main then rerun make_paper_equations.py,pending after T1
  T3,code,panel_deposits.py double-read fix AT THE NEXT FULL REBUILD only,deferred (user 10-05)
  T4,code,trace the 0.8% post-2016 ESTBAN shortfall inside the panel build,open
  T5,code/cluster,next SLEEP_STEP=entry upload: ship the edited sleep_ident_entry_dynamics.py (it imports utils/window.py),note
  T6,code,D-firm g fallback (31.0% of D entrant-quarters use the national median),optional
  T7,whoever owns the motivation memo,slide.motiv.entrygap numbers moved (n 202 -> 198); check any text quoting them,check
  T8,Pedro,delete superseded DIAG_ENTRY_ESTBAN outputs (*_full*, *_from2015Q4*); my rm was declined,ask
```

**T1 (author).** The caption of `fig:phi_entry_dyn_fixed_phi` has two typed phrases that no longer hold.
- It reads `The \pn[w]{phi.entry.drop_robustness.n_dropped.D} $\mathrm{D}$-firm entrants whose plateau share ... have no defined path ... so they appear only on the right panel`. The count is now one, so the plural is wrong.
- It also says the B-firm censored entrants are "of 2023Q3". They are now 2022Q3 to Q4, and there are also 3 D-firm censored entrants. The new key `phi.entry.drop_robustness.n_censored_both.D` (three) exists in the macros.
- The caption should also state the sample rule: entries dated 2016Q1 to 2024Q4, paths observed through 2024Q4.
- Acceptance: every `\pn` key resolves (no red ??), and the caption says nothing about a particular quarter that the data no longer support.

Replacement text (Pedro asked for it; he pastes it himself). In V_Main.tex line 755 (the caption), find:

```latex
The \pn[w]{phi.entry.drop_robustness.n_dropped.D} $\mathrm{D}$-firm entrants whose plateau share is within $10^{-6}$ of their entry share have no defined path under the plateau normalization, so they appear only on the right panel. Including them moves the $\mathrm{D}$-firm median by at most \pn[2]{phi.entry.drop_robustness.max_abs_diff} and the width of its 95 percent band by at most \pn[2]{phi.entry.drop_robustness.max_abs_width_change}. The \pn[w]{phi.entry.drop_robustness.n_censored_both.B} $\mathrm{B}$-firm entrants of 2023Q3 are not yet observed through quarters 10 to 12 and are excluded from both panels.
```

Replace with:

```latex
Entries are dated within the 2016Q1--2024Q4 sample, and share paths are observed through 2024Q4. $\mathrm{D}$-firm entrants whose plateau share is within $10^{-6}$ of their entry share (\pn{phi.entry.drop_robustness.n_dropped.D} here) have no defined path under the plateau normalization, so they appear only on the right panel; including them moves the $\mathrm{D}$-firm median by at most \pn[2]{phi.entry.drop_robustness.max_abs_diff} and the width of its 95 percent band by at most \pn[2]{phi.entry.drop_robustness.max_abs_width_change}. The \pn[w]{phi.entry.drop_robustness.n_censored_both.B} $\mathrm{B}$-firm and \pn[w]{phi.entry.drop_robustness.n_censored_both.D} $\mathrm{D}$-firm entrants whose quarters 10 to 12 fall after 2024Q4 are excluded from both panels.
```

All five keys exist in `paper_numbers_macros.tex` (18:39): `n_dropped.D` = 1, `max_abs_diff` = 0.037, `max_abs_width_change` = 0.043, `n_censored_both.B` = three, `n_censored_both.D` = three. Once the caption states the sample rule, line 741 needs no change; a sentence there is optional and not drafted.

**T3 (code, deferred).** `panel_deposits.py` `load_estban_raw_pre2016()`, `sorted(glob("*.CSV") + glob("*.csv"))`. On Windows this lists the 36 pre-2016 files twice, so every pre-2016 ESTBAN balance is doubled. The fix must go in TOGETHER with removing the demand repair file's 2015Q4 halving (`utils/demand_repair.py`, `panel_demand_repair.py`); applied alone, it halves 2015Q4 twice. Measured panel/raw national ESTBAN deposits: 1.985 to 1.986 in 2015, 0.992 from 2016Q1.

**T4 (code).** After 2016 the panel's ESTBAN total is 0.817% below the raw files. `ESTBAN.csv` equals the raw files (2016Q1 R$ 1,529.9 bn both), and raw rows without a municipality carry no deposits. The loss is therefore inside `panel_deposits.py` / `panel_market.py`. Not traced.

**T5 (cluster).** `sleep_job.sh` L.260 runs `--compute-only --routine-curves ...` with the defaults, so the window rule is now active there too. If a cluster D6 run is ingested, it must carry `window_q` in `d6_meta.json`; otherwise it is the old rule.

## 4. Honesty flags (keep visible)

- The quoted share moved: `phi.entry.point` 0.917 to 0.914; the conclusion's `\pn[2]` now prints 0.91 (was 0.92). Model curves (0.983/0.981) still sit above the B interval; the D interval reaches the top of the grid.
- The macro refresh also changed three `sleep.ssel.E3.*` keys (0.983 to 0.984, 0.978 to 0.980, 0.983 to 0.984). These come from another stream's `make_state_vector_table.py` output, not from this work, and V_Main does not use them.
- The double read was already traced on 10-05 (demand repair). This session re-found it; the research note now says "known since October 5".
- Data-source effect: raw ESTBAN (institution level, every ESTBAN reporter in the denominator, including the 19 ESTBAN-reporting conglomerates the panel books as D) gives +0.006 over the panel at the same fixed g. The accrual path effect (per-entrant vs fixed g) is below 0.001.
- g is constructed, not estimated. 81.1% of B entrants have no own cell in their entry quarter, because a cell's first quarter has no lagged rate.
- Pre-2016 D shares are distorted by the double read: ESTBAN-sourced D firms are doubled, IF.Data D firms are not. This is irrelevant under the window rule.
- The demand-repair stage C is in progress in another session. Do not rebuild the panel in the middle of it.

## 5. Open questions for Pedro

1. Paste the caption replacement (T1)? A sample-rule sentence at L.741 is optional once the caption carries it.
2. Cite the raw-ESTBAN check in the paper (it would be one sentence plus a footnote), or keep it as a research note only?
3. When to schedule the full rebuild that carries T3?
4. Delete the superseded `DIAG_ENTRY_ESTBAN` outputs from the earlier samples (T8)?
