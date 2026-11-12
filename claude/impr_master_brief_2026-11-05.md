# Master brief, 2026-11-05: every open item, one document

**To:** the IMPR session (the main agent). **From:** the main thread, for Pedro; written 11-04 23:05, consolidated 11-05 00:10. **Scope.** On 11-05 Pedro sends you one command and you run everything below, spawning one subagent per thread (Kappa, Pix, tables) briefed from the thread section here. The four thread handoffs (`impr_handoff_2026-11-03.md`, `kappa_handoff_2026-11-03.md`, `pix_handoff_2026-11-04.md`, `tables_handoff_2026-11-04.md`) were deleted on 11-05 at Pedro's instruction; this brief is self-contained. Their rules are in Section 0, their approved script changes (with LF md5s) in each thread section, the I10 map in Section 2, and their decisions in `SHARED_MEMORY.md`. `SHARED_MEMORY.md` entries 11-04 19:00 to 23:00 are the record of last evening. Nothing below edits `V_Friction.tex`.

## Update 11-12 (main thread): the reply in force is `MAIN_REPLY_2026-11-12.md`

The update lines of 11-05 to 11-11 that stood here pointed to `MAIN_REPLY_2026-11-05.md`. That file was closed and deleted on 11-12 at Pedro's request, and its history is in the log of `SHARED_MEMORY.md`. Read `MAIN_REPLY_2026-11-12.md` for the work in order. The sections below are the brief of 11-05, kept for the rules of Section 0 and the map of the pipeline. Where they describe tasks, most are done or parked.

## 0. Rules that carry over

- `V_Friction.tex`, the decks and the design notes are edited by Pedro and the main thread only; agents deliver code, outputs and reports, and change summaries for the paper's generators.
- Pedro owns git; agents do not delete files: give `Remove-Item -LiteralPath` lines. Raw files stay untouched. New scripts under `diagnostics/` (the Kappa agent's under `diagnostics/` or `kappa/`), production scripts only through the approved changes.
- Python `C:/venvs/egan/Scripts/python.exe -I` with `MPLBACKEND=Agg`. One heavy process on the machine at a time: your pipeline reruns, the Kappa job, step 7 and the K1 run do not overlap; announce start and end lines in your reply. Read `market_panel.parquet` through pyarrow by column subsets, never through `load_panel_cached`.
- Never guess a schema or a file layout: read the file, or ask Pedro for its documentation. Timestamp every update; "wait" means stop.

- One panel md5 per run, posted as the end line; every output row carries it. The Kappa agent's step 7 and the K1 inputs pin the snapshot `52ea6495` (its read-only copy) until K1 is restarted on the final panel.
- No script edit without a one-paragraph plan and Pedro's OK, except those already approved (listed per thread); LF md5 of every edited script in the reply.
- Light estimation locally; the cluster chain (I2) only on Pedro's launch lines.
- Replies: `IMPR_REPLY_2026-11-05a.md` for your work, summarizing the subagents' state; subagents in their threads' dated files (`KAPPA_REPLY_2026-11-04c.md`, `PIX_REPLY_2026-11-04e.md`, `TABLES_REPLY_2026-11-04c.md`). Delete nothing in `claude/`; the main thread does.
- Scratch outside `Rout` and the paper folder: give Remove-Item lines.

## 1. Pedro decides first

```
decisions[3]{id,question,numbers}:
  D1,"imputed time-deposit rate tails: (a) rate/CDI outside 0.5 to 1.5 goes to the carry, or (b) a band in percentage points above the CDI (+3, +4 or +5 pp, and the band below)","your reply (e) of 11-04, I15: the excess is a level of 4.5 to 11 pp a year above the CDI at foreign and wholesale banks and mid-size fixed-rate issuers, all from calibrate's strip; the ratio rule bites in 2018 to 2021 because the CDI was low; I16 below gives the counts per band"
  D2,"the Pix placebo: replace the single shuffle in diag_p_shifters.py by 99 or more, for both specifications, with the placebo median and the share at or above the real F carried into p3_numbers.json and into the paper key pix.stakes.fs.f_placebo","Rout/p3_placebo_shuffles_0b9385b5.json: on 0b9385b5 the file's one draw gave 484 against a real F of 493; 20 shuffles give a median of 158 (range 30 to 594), 35 net of structure; the cost shifters beat 19 of 20"
  D3,"launch I2 (the cluster chain) on the panel D1 produces, with the restaged package and the seven launch lines","impr_handoff_2026-11-03.md, Update 16:00 and reply (e): the package of 20:02 on 0b9385b5 (248 files) is replaced by I17's"
```

## 2. IMPR (you)

```
tasks[5]{id,what,after}:
  I16,"read-only: counts of imputed B conglomerate-quarters and their type-4 weight by year for rule (b) at +3, +4, +5 pp above the CDI (and the symmetric band below), beside rule (a)'s 378 weighted quarters; post for D1",now
  I17,"apply D1's rule to imputed rates (same flag discipline as I14: its own calibrate flag and fallback code, so the report says which quarters the rule removed); rerun from calibrate; post rows changed by bank, type-4 weight by source, tails left, and the new panel md5 as the end line; restage the I2 package on it with sha256 and the launch lines",D1
  I2,"on D3: Pedro launches; you watch the gate jsons and report",D3 and I17
  I10,"productionization: K0's load_demand checks the frame's metadata md5 (today mtime); K0's panel pin as an environment override and the loader rules of 11-03; the Bradesco 2016 observation; the map below; not before I17 and the I2 launch",I2
  B-registry,"11-13: registry requests 025399 (lambda and omega by bank, early against at-maturity redemptions) and 025440 (CDB issuance by issuer and investor state: issuance, stock, mean rate as a percent of the CDI) arrive; a BB-type handoff then; with 025440, repeat T10b (B versus D rates against the panel) on the registry rates; nothing now",11-13
```

Basis for I17: `panel_cosif_calibrate.py` LF `c1d22980`, `panel_deposit_rates.py` LF `db921580` (after I14); panel `0b9385b5410b5cb6ae4c884c32e4ef46`; I15's `i15_imputed_tails.csv` in your scratch is the universe (520 quarters, 378 with deposits).

The I10 map, as written on 11-03 (confirm against the code when you get there):

The side scripts that found and sized the defects are not where the pipeline will live. Once the other agents are done (B7 landed, K1 run, P3 rerun), you reorganize: every adopted object gets a home in the panel build or in the estimation routine, with its raw inputs registered (source log, `DATA_RAW` folder, upload manifest where the cluster needs them) and the diagnostic retired or kept as a test under `diagnostics/`. A change summary per script before anything moves; the runner (`exhibits_post_cluster.py`, `pipeline_all.sh`) gains the steps. The map, to be confirmed against the code when you get there:

```
i10[10]{object,now,home}:
  Judicial deposits J_jt,"diagnostics/diag_judicial_deposits_cosif.py, diag_judicial_deposits_share.py; Rout/judicial_deposits_share.csv; compiled DFP extracts","a panel step (panel_judicial_deposits.py) that builds J_jt from DATA_RAW/issuers/judicial_deposits/compiled/ and the Cosif 2025 titles, feeding utils/judicial_deposits.py; the diagnostics become its tests"
  DFP funding-expense benchmarks,"coding-thread scratch (BB) and B7's compiled file","DATA_RAW/issuers/funding_expense/ with a loader in panel_cosif_calibrate.py (D6)"
  Savings yields,"I8 scraper, new","scrape_bcb_savings_yields.py in the scraping step; SGS_RAW file; utils/macro_rates.py reads it"
  Rate fallbacks,"Rout/rate_fallback_audit.csv (audit)","the guard and the carried rate inside panel_deposit_rates.py and panel_cosif_calibrate.py (D1); the audit stays as a diagnostic"
  Market size,"applied in sleep_demand_prep_link.py (I1) and the switch-cost copy (D5)","done; no further move"
  Net-flow objects K0,"diagnostics/diag_k0_netflow_objects.py","make_netflow_objects.py in the estimation routine, writing the k0_* files as pipeline outputs"
  Shifters P1 to P3,"diagnostics/diag_p_shifters.py","make_kappa_shifters.py in the estimation routine, after the panel and the CCS steps; P3 depends on the panel md5"
  K2 simulation and K1 regression,"Kappa agent's scripts, when adopted","a kappa/ module in the estimation routine, with its tables under the paper's table scripts and \\pn keys only when Pedro says"
  Booking checks (BB3, B1),"diagnostics/diag_bb_rende_facil_booking.py, diag_santander_sweep_booking.py","stay diagnostics; their verdicts live in the design note"
  Pix roster and 2019 check (R1, R1b),"Rout/pix_participation_*; diagnostics/diag_pix_mandate_2019.py","the roster becomes a panel side table if the awareness margin uses it; else stays"
```


## 3. Kappa subagent

State at 20:43 (reply (b)): the D3 job (steps 3 to 6, PID 25148) was running, end about 01:00; Pedro's machine shut down at 20:50, so assume the job died. Approved and done: `read_menu`'s closed-form fallback keyed on the frame's metadata md5 (`diag_k2_netflow_simulation.py` LF `55787e72`), `--panel PATH` in `diag_k2_nls_simulation.py` (LF `f517111f`), the read-only snapshot `...\904ee55d-75e4-4e82-a3ca-87ddc3397f97\scratchpad\panel_52ea6495\` (panel `52ea6495`, side file `861a7e24`).

```
tasks[5]{id,what,after}:
  K-resume,"list which fits of steps 3 to 6 are recorded in Rout/k2_nls_numbers.json; resume from the first missing one with the same warm starts (step-2 per-pair fe row, one start each); the stored-world K2b stays skipped; post the finish line",now
  K-tables,"the step 3 to 6 tables and the NLS sections (step 2, the decomposition, K2b, steps 3 to 6) in Rout/k2_report.md; the step-3 reading so far: no break 0.507 (savings), the 30 percent time-deposit break sends the full-sample savings pair to its bound while the sample without the scaled bank keeps 0.471",K-resume
  K-key,"decision A (Pedro, 11-04 20:48): write the money-weighted awake share 0.0177 to k2_numbers.json at pool_fate_kappa_base.awake_money_share_of_deposits with a _definition string (formula, weights, 36 quarters, panel d01b8bae); nothing else in that file changes",K-resume
  K7,"decision B: step 7 with --out Rout\\k2_step7_52ea6495\\ (no code change), on the snapshot: K0_PANEL_MD5=52ea6495 DEMAND_CHARS_FILE=<snap>\\demand_chars.parquet DEMAND_PREP_DIR=<interim> python -I diagnostics/diag_k2_nls_simulation.py --steps 7 --panel <snap>\\market_panel.parquet --ctx <snap>\\k2n_ctx_52ea6495.pkl --out Rout\\k2_step7_52ea6495\\ ; <interim> = ...\\21a30b32-...\\scratchpad\\rebuild_1104\\interim_q1b\\demand_prep\\ (frames of 19:41, metadata 52ea6495, index frame 610,459 rows); about 40 minutes plus the build; machine quiet",K-resume
  K1-spec,"write the K1 plan on I17's panel, from the design note (kappa_design_2026-11-02.md, Sections 3.5 to 3.7 and 5 Group C): per-pair kappa-tilde with the savings pair as headline, total-pool weighting, effects in levels solved by market, Levenberg-Marquardt from two starts, control function for the spread from the own cost shifters, Pix before-after and the dose instrumented by 2019 mobile and 4G/5G access, wild cluster bootstrap by conglomerate; the inputs it needs (P3's stakes on the same md5, the sweep registry dates, omega and lambda as today); the frame's row count and run-time estimate; no heavy run before Pedro's OK",I17
```

## 4. Pix subagent

State at 21:40 (reply (d)): P3 to P5 rerun on `0b9385b5` (stakes median R$42.5, F 493, 370 net of structure; advertising 0.086 rest, 0.063 big five n.s.; IV 0.204, AR [0.053, 0.477]; group placebo −0.770); `diagnostics/diag_p3_numbers.py` writes `Rout/p3_numbers.json` keyed by panel md5. The thread's rules: pyarrow column subsets, md5 in every row, Remove-Item lines for scratch.

```
tasks[2]{id,what,after}:
  P-shuffle,"if D2 is yes: in diag_p_shifters.py replace the single placebo by 99 or more shuffles for both specifications (plain and net of structure), report the placebo median and the share at or above the real F, carry them into p3_numbers.json; post the LF md5; no change to any column of the stakes CSV",D2
  P8,"on I17's md5: diag_p_shifters.py --only p3; diag_p3_numbers.py; diag_p4_openings_advertising.py (about 7 minutes); each note gains one column; if K1 asks for P3 on the 52ea6495 snapshot, add a panel-path option to diag_p_shifters.py on request only",I17
```

## 5. Tables subagent

State at 19:41 (reply (b)): collectors `collect_tax_clock`, `collect_k2b`, `collect_k0`, `collect_k2_world`, `collect_pix_adv`, `collect_pix_stakes` in `make_paper_numbers.py`; generators `make_tax_clock_table.py`, `make_k2b_pairs_table.py`; `tab_tax_clock.tex`, `tab_k2b_pairs.tex` in the paper folder; 136 new keys, 64 `desc.*` moved with the rebuilt descriptive tables (Pedro to reread the B time-deposit spread sentence). Scratch `...\fed7f088-...\scratchpad` holds `macros_before.tex` until Pedro has seen the `desc.*` change.

```
tasks[3]{id,what,after}:
  T7,"collect_pix_stakes reads Rout/p3_numbers.json['panels'][md5] (md5 = the stakes CSV's; floats), not the markdown table; register kappa.k2.world.awake_share from k2_numbers.json (format 3; the paper prints it with p1); read step 7's k2_rc.csv from Rout\\k2_step7_52ea6495\\ when it exists; if D2 is yes, f_placebo quotes the shuffle median and a new key pix.stakes.fs.placebo_share_above carries the share at or above the real F",K-key and K7 and P8 (each part as its input lands)
  T8,"after every change above: python make_k2b_pairs_table.py; python make_paper_numbers.py; post the keys that changed with old and new values (the paper quotes kappa.k2b.*, kappa.k0.*, kappa.k2.world.*, pix.adv.*, pix.stakes.*, clock.t3.*), 0 MISMATCH, no other key changed",T7
  T9,"when K1 runs: a generator for its table (by pair: level, change, dose, with the bootstrap intervals) and keys kappa.k1.*; wait for the K1 plan",K1-spec
```

## 6. The paper (Pedro, not an agent)

`claude/paper_edits_2026-11-04.tex`: W5 (content: the second line of the utility and the Section 6 body), W7 (three appendices), F1 to F6 (Figure 2 with panels a, c, d). After applying: compile with biber; `python make_paper_equations.py` for the deck macros. T8's list says which quoted keys moved. The one red literal left is `kappa.k2.world.awake_share` (1.8), replaced by `\pn[p1]{kappa.k2.world.awake_share}` once T7 lands.
