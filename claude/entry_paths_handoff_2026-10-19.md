# Handoff: Figure 2 entry paths (window rule, raw-ESTBAN check, accrual g)

**Date:** 2026-10-19, 18:12. Replaces `entry_paths_handoff_2026-10-12.md`, which was deleted.

**Scope:** Figure 2 of V_Main (`fig:phi_entry_dyn_fixed_phi`), the median share paths of entrants, and the entry-implied sleeping share (`phi.entry.*`).

**Rule:** V_Main.tex is edited by the author only. The deck is never edited.

**Basis:**

```
files[7]{file,role,saved}:
  sleep_ident_entry_dynamics.py,Figure 2 generator (window rule 10-12; legend layout commit a968b798 10-16),10-16 15:53
  market_panel.parquet,final panel (main thread's rebuild),10-15 22:35
  DEMAND_PREP/est3|est4/national_phi_t.csv,model-curve phi (instrument RivalSpread_x_Tech),10-18 16:30/16:35
  DIAG_PHI_SEPARATION/d6_*,Figure 2 outputs on the final panel,10-19 18:06
  Drafts/Deposit Competition/fig_entry_dynamics_{a,b}.*,Figure 2 panels,10-19 18:06
  Drafts/Deposit Competition/paper_numbers_macros.tex,refreshed by make_paper_numbers.py,10-19 18:07
  Drafts/Deposit Competition/fig_slide_motiv_entry_by_gap*.pdf,deck chart (fidelity 2.2e-16 vs d6_entry_paths.csv),10-19 18:07
```

## 1. What the code does

| Element | What it does | Where |
|---|---|---|
| Sample rule | Entries dated 2016Q1 to 2024Q4 (`utils/window.py`); quarters after 2024Q4 cut; earlier quarters only show the entrant was absent before. On by default; `--no-window-only` uses every panel quarter | `WINDOW_Q0/Q1`, `load_event_frames()` |
| B-firm share | Entrant's deposits (types 1, 2, 4) over all locally booked deposits in the MCA. Local panel rows are all ESTBAN and D rows sit at CODMUN 0, so the share is conditional on B firms | `load_event_frames()`, `build_events()` |
| Branch screen | Deposits two or more quarters before the first branch, judged only from the first quarter the branch data cover (2016Q1) | `load_branch_entry()` (`attrs["q_cover0"]`), `build_events()` |
| Accrual g | Per entrant and quarter, deposit-weighted `gross_return_lag` over types 1, 2, 4 of its own cells; fallbacks: conglomerate cross-market mean, then national median, then flat carry. Constructed, not estimated | `build_g_paths()` |
| `gross_return_lag` | 1 + r^f(t-1) - spread(t-1). Type 1 pays 0; type 2 pays TR + 0.5%/month (Selic target above 8.5%) or TR + 70% of the target; type 4 pays the conglomerate's COSIF implicit rate, with CDI above 3 x Selic | `sleep_demand_prep_*.py`, `panel_deposit_rates.py` |
| Implied phi | Grid of 1,400 points on [0.50, 0.9985]; minimizes the squared distance of median model vs median data over h = 1 to 9 (plateau normalization); 400 bootstrap resamples of entries | `implied_phi()` |
| Model curves | Mean national phi_t over 35 quarters per routine, on each entrant's own g path | `phi_vintages()`, `vintage_curve()` |

## 2. Results (final panel, 10-19)

```
results[5]{object,value,ci95,n}:
  phi_entry B,0.914,[0.891 0.937],198
  phi_entry D,0.986,[0.888 top of grid],59 main / 60 alt
  phi*g B,0.926,,198
  model phi E3 (III),0.986,,
  model phi E4 (IV),0.985,,
```

- The B-firm result is identical, to four decimals, to the 10-12 run on the 08-26 panel. The final panel did not change these entries' deposits.
- Only `phi.entry.model.E3/E4` moved, from 0.983 and 0.981, because the sleepiness stage now uses the rival-spread instrument. V_Main does not quote these two keys.
- Kept B entries: 201 (2016:36, 2017:15, 2018:23, 2019:45, 2020:36, 2021:29, 2022:17). Of these, 198 are plotted; 3 entrants from 2022Q3 to Q4 are not observed through h = 10 to 12. D: 63 kept, 3 censored (2022Q4), 1 dropped by the plateau guard.
- The branch screen drops 1 entry.
- History: the old rule (entries from 2014, paths through 2025Q4) gave 0.917 [0.894, 0.941], n = 202. Its branch screen dropped 125 entries, 124 of them 2014Q1 to 2015Q3 entries with no branch data. The archive of those outputs was deleted by the user on 10-13.
- Raw-ESTBAN check (10-12, same rule, 08-26 inputs): 0.921 [0.899, 0.942], n = 211; median paths within 0.015 of the paper's. Paper entries at a fixed g = 1.011: 0.915.
- The model curves (0.986, 0.985) lie above the B interval and inside the D interval.

## 3. Tasks

```
tasks[5]{id,agent,deliverable,status}:
  T1,Pedro,V_Main caption L.755 find/replace below; then recompile + make_paper_equations.py,pending
  T2,other agent,report_entry_paths_estban.py reads the deleted _D6_PREWINDOW_20261012 archive; regeneration fails,owned elsewhere
  T3,code,panel_deposits.py double-read fix (with removing the 2015Q4 halving) at the next full rebuild,deferred
  T4,code,0.8% post-2016 gap between the panel's ESTBAN total and the raw files,open
  T5,code,31% of D entrant-quarters use the national median g,optional
```

**T1.** In the caption at V_Main.tex line 755, find:

```latex
The \pn[w]{phi.entry.drop_robustness.n_dropped.D} $\mathrm{D}$-firm entrants whose plateau share is within $10^{-6}$ of their entry share have no defined path under the plateau normalization, so they appear only on the right panel. Including them moves the $\mathrm{D}$-firm median by at most \pn[2]{phi.entry.drop_robustness.max_abs_diff} and the width of its 95 percent band by at most \pn[2]{phi.entry.drop_robustness.max_abs_width_change}. The \pn[w]{phi.entry.drop_robustness.n_censored_both.B} $\mathrm{B}$-firm entrants of 2023Q3 are not yet observed through quarters 10 to 12 and are excluded from both panels.
```

Replace with:

```latex
Entries are dated within the 2016Q1--2024Q4 sample, and share paths are observed through 2024Q4. $\mathrm{D}$-firm entrants whose plateau share is within $10^{-6}$ of their entry share (\pn{phi.entry.drop_robustness.n_dropped.D} here) have no defined path under the plateau normalization, so they appear only on the right panel; including them moves the $\mathrm{D}$-firm median by at most \pn[2]{phi.entry.drop_robustness.max_abs_diff} and the width of its 95 percent band by at most \pn[2]{phi.entry.drop_robustness.max_abs_width_change}. The \pn[w]{phi.entry.drop_robustness.n_censored_both.B} $\mathrm{B}$-firm and \pn[w]{phi.entry.drop_robustness.n_censored_both.D} $\mathrm{D}$-firm entrants whose quarters 10 to 12 fall after 2024Q4 are excluded from both panels.
```

All five keys resolve in the 10-19 macros. Checked 10-19 18:00: V_Main.tex still has the old text.

## 4. Honesty flags

- The B result was recomputed on the final panel and the φ̂ inputs of 10-18. It did not change.
- `entry_paths_estban_report.pdf` (10-12) is still correct for B, D and raw ESTBAN, but its model-curve values (0.983/0.981) are now stale (0.986/0.985).
- The raw-ESTBAN check was not rerun on the final panel. It reads the raw files plus the panel's municipality-to-MCA map only.
- Pre-2016 ESTBAN levels are doubled in the panel. No path uses them under the window rule.
