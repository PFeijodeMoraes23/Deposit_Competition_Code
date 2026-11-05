# Handoff: the switching-cost design (Kappa agent, new session)

**To:** the Kappa agent. **From:** main thread, for Pedro. **Written:** 2026-11-03, 21:00.

**Scope.** You run the empirical side of the switching-cost design: first a simulation that checks what the design's regression measures (K2), then the regression itself on the rebuilt panel (K1). The design is written out in `Drafts/Deposit Competition/kappa_design_2026-11-02.md` (twenty-second version, 11-03, 40 pages; the PDF beside it is the rendered copy; assumption labels run A1 to A7: A1 mean utility, A2 dominant stay, A3 quality weights, A6 the separable scale, A7 the outside pool). **Since the twenty-second version there is no linear regression: the estimating equation (33) is nonlinear in the gap and estimated by NLS or PPML with a control function for the spread, and the parameters (34) are read directly. Anything below that says "slope", "ratio of slopes" or "OLS" is superseded by this update.** Read it before anything else: Sections 2 (the model), 3 (why the reduced form), 4 (the derivation, Steps 1 to 5), 6 (the tests, Groups A to C) and 7 (the order of work); Section 10 (the record appendix) has every task already run with its result, and K0 there is the data you start from. Reply in `claude/KAPPA_REPLY_2026-11-03.md`, timestamped; newest section first.

## Update 11-04, 20:50: decisions A and B; the machine shuts down tonight

Your reply (b) read. **Decision A (Pedro, 20:48):** definition (1), the money-weighted share, 0.0177; write it to `k2_numbers.json` at `pool_fate_kappa_base.awake_money_share_of_deposits` with the `_definition` string (formula, weights, the 36 quarters, panel `d01b8bae`) at the finish line; the paper and the design note change 1.2 to 1.8 percent. **Decision B:** option 1, run step 7 with `--out Rout\k2_step7_52ea6495\`; `Rout/k2_nls_numbers.json` keeps the `d01b8bae` record; the tables agent reads `k2_rc.csv` from the subfolder. No code change. The step-3 partial results and the 01:00 end are noted.

**Pedro's computer shuts down tonight (20:50).** If the job is killed, do not restart it blind: when the machine is back, say in `KAPPA_REPLY_2026-11-04c.md` which fits of steps 3 to 6 are recorded in `k2_nls_numbers.json`, and resume from the first missing one with the same warm starts; then the awake-share key and step 7 as above. If the job completes before the shutdown, post the finish line as planned.

## Update 11-04, 20:25: the interim frames exist; nothing else is open on our side for step 7

The coding thread's reply (e) names them: `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\rebuild_1104\interim_q1b\demand_prep\`, ready 19:41 on `52ea6495`: `demand_1_spec_12.parquet` (620,701 rows), `demand_2_spec_12.parquet` (619,595), `demand_3_index_spec_12.parquet` (610,459), `demand_4_index_time_spec_12.parquet` (616,779), each with `demand_chars.market_panel_md5` = `52ea6495`; fits in `interim_q1b\est1` to `est4` with the new centering `f74a37d3`. The row count differs from the production frame's 630,522, and your fallback is keyed on the metadata in any case. I14 has since landed (`0b9385b5`, 10 Itaú and Banrisul quarters re-carried); it is for the cluster package and the Pix reruns, not for you. So: your finish line, then the `--panel` build on your `panel_52ea6495` copy with `DEMAND_PREP_DIR` at the folder above, then step 7. If you were waiting on anything else, say what in `KAPPA_REPLY_2026-11-04b.md`.

## Update 11-04, 20:10: one number for the paper's keys (small, with your finish line)

The tables agent registered the K2 numbers the paper quotes from `k2_numbers.json` and `k2b_pix_change.csv`. One is in no file: awake money as a share of deposits in the kappa-base world, the 0.012 your step-2 reply and `k2_report.md`'s reading gave ("awake money is 1.2 percent of deposits"). With your finish line, write it to `k2_numbers.json` under `pool_fate_kappa_base.awake_money_share_of_deposits` (say how it is computed: awake movable money over total deposits, money-weighted, which quarters), so `make_paper_numbers.py` can read it as `kappa.k2.world.awake_share`. Nothing else changes.

## Update 11-04, 19:35: option 1 approved; the fallback keyed on the frame's metadata; step 7 reads your copy

Your 19:17 message read; the read-only copy of the panel and its side file at `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\904ee55d-75e4-4e82-a3ca-87ddc3397f97\scratchpad\panel_52ea6495\` (panel `52ea6495`, side file `861a7e24`, times kept) is the file step 7 reads, so I14 can overwrite the production path whenever it is ready.

**Option 1, approved (Pedro, 19:30).** Add `--panel PATH` to `diag_k2_nls_simulation.py`. When given, it points K0's `load_panel` at that file for the build and records the path and the md5 read in the numbers JSON; when unset, nothing changes. The side file through `DEMAND_CHARS_FILE`. K0 and the old K2 script stay untouched. Post the LF md5 of the edited script.

**Amendment to item 1, approved with it.** Do not key the closed-form fallback in `read_menu` on the row count. Key it on the frame's metadata: when `demand_chars.market_panel_md5` in the frame equals the panel md5 the stored binary was built on (the 10-27 panel, `d01b8bae`), use the binary with the $10^{-8}$ check; otherwise take the closed form and record the source. Same function, same test as at 18:55 (the scratch binary one row short becomes a scratch frame whose metadata carries another md5). This removes the 630,522-row coincidence you flag.

Everything else as at 19:15: the interim frames from the coding thread's sandbox (`...\rebuild_1104\interim_q1b\demand_prep\`, end about 20:00 to 20:30, location to be posted by the coding thread), metadata md5 `52ea6495` checked before the build, step 7 after your finish line (about 21:00). Reply in `KAPPA_REPLY_2026-11-04b.md`.

## Update 11-04, 19:15: item 1 accepted; step 7 pins `52ea6495`, not a later md5

Your 18:55 message is accepted (the closed-form fallback with its record, LF md5s `eb29157c` and `5317999c`, the tests). The coding thread's end line is in: the panel is **`52ea6495c9195ef38b83bc40fb2014fa`** (`K0_PANEL_MD5=52ea6495`), carrying Q4 and I12. The interim frames are being written on it now (started 18:42, end about 20:00 to 20:30) under the coding thread's sandbox `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\rebuild_1104\interim_q1b\demand_prep\` (`demand_3_index_spec_12.parquet` and the others; metadata key `demand_chars.market_panel_md5` should read `52ea6495`). A further small rate change (I14, a few Itaú and Banrisul semesters) will produce one more md5 for the cluster package; **step 7 does not wait for it** and pins `52ea6495`. Everything else as at 18:40.

## Update 11-04, 18:40: K2b accepted; item 1 approved; step 7 after your finish line and the coding thread's end line

Your 18:22 reply (`KAPPA_REPLY_2026-11-04a.md`, removed after processing) is accepted and recorded in the design note: the gate; the savings pair as the one headline (level, change and dose in both worlds); $\alpha$ and $\alpha^P$; the checking pair not pinned down and the time-deposit pair's dose unreliable, so neither is a second dose estimate; D2 settled by the divergence of the own-savings row.

**Item 1, approved (Pedro, 18:35):** edit `read_menu` as you propose. When the binary's row count differs from the frame's, take $\delta$ from the closed-form inversion the function computes and record the source in the outputs; when they match, keep the binary and the $10^{-8}$ check. Post the LF md5 of the edited script. **Item 2:** noted; the md5 check in `load_demand` goes to the coding thread's I10 list, not to you. Read the interim frame's metadata yourself, as you say, and run step 7 only on the md5 the coding thread's end line names (the I12 rerun, started 18:03, replaces `45dbfc58`).

**Order unchanged:** your finish line (steps 3 to 6, about 21:00), the coding thread's reply naming the sandbox and the final md5, then the input build and step 7 (about 40 minutes). Reply in `KAPPA_REPLY_2026-11-04b.md` with the step 3 to 6 tables, the `k2_report.md` sections and step 7.

## Update 11-04, 16:30: D3 amended, skip the stored-world K2b; step 7 as you describe

Your 16:26 message read. D3 amended: skip the stored-world K2b. Step 2 already shows that world fails (five fits at the cap, $\kappa$ at the edge, the A6 result), and K2b changes weighting, not the world. Stop the job when it reaches that block and post the finish line then. Step 7 as you describe it: rebuild the inputs with `K0_PANEL_MD5=<final md5>` once the coding thread's reply names the interim frames, then run. Steps 2 to 6 on the 10-27 base data are fine for what they test. Reply in `KAPPA_REPLY_2026-11-04a.md` after the ideal K2b, as planned.

## Update 11-04, 16:00: the rebuilt panel and the frames for step 7 and K1

The panel rebuild landed at 15:04 (md5 `70241f4e`); a 15-minute rate fix reruns from the rates step and posts a final md5 in `impr_handoff_2026-11-03.md`: pin `K0_PANEL_MD5` to that one. The demand frames (the $\hat\phi$ and $\widehat M$ your step 7 and K1 read) are being rebuilt locally as interim frames, with the panel md5 in their metadata, in a sandbox the coding thread names in its next reply; K0's `load_demand` accepts them by md5. Step 7 after your finish line and that reply.

## Update 11-04, 13:45: the rebuild runs in parallel with your job; step 7 after both

Your 13:14 note is read: end about 20:00; the fe-world per-pair K2b (savings level 1.006, change $-0.501$; checking 0.48 and $-0.22$; time deposits 0.95 and $-0.49$; $\alpha^P=0$) is recorded in the note. Pedro (13:45): the coding thread starts the panel rebuild now, in parallel with your job, rather than at your finish line; it is I/O and single-threaded, your job is one CPU-bound process. Nothing changes for you except that step 7 waits for both your finish line and the rebuild's end line in `impr_handoff_2026-11-03.md`, whichever comes later. Reply as planned in `KAPPA_REPLY_2026-11-04a.md` after the ideal-world K2b.

## Update 11-04, 11:30: your 03b reply accepted; decisions D1 to D4; the run order

Accepted in full: the solver (effects in levels, exact effect step by market, Levenberg-Marquardt on the projected Jacobian, two starts), the unit test, the attribution of the two PPML failures to the solver, the step-2 table, the per-pair finding, K2b. Reply next in `KAPPA_REPLY_2026-11-04a.md`.

```
decided[4]{item,decision}:
  D1,"(a): the per-pair form; the savings pair is the gate object and the headline; the cross-pair comparison is reported as a measure of A2's error and of the checking scale's weak identification, not as a test of kappa"
  D2,"the total-pool weighting (each row divided by T = 1 + Q1 + Q4 + R1 + R2 + R4 + O) is the estimator's form; own-savings normalization is the robustness row"
  D3,"run now: the stored-world K2b (before-after and dose) and steps 3 to 6, on the per-pair form; write k2b_pix_change.csv and the k2_* files; add the NLS sections to Rout/k2_report.md"
  D4,"step 7 runs last, after D3 and after the coding thread's panel rebuild has ended (it is announced in impr_handoff_2026-11-03.md); not while either runs: one heavy process on the machine, and the 14 GB read waits for a quiet machine"
```

**Order on the machine.** Your D3 job first; the coding thread's rebuild starts when it ends (it reads this file for your "done"), so post a line here with the finish time; step 7 after the rebuild's end line. The gate for D3 stays: if the savings pair does not recover the change within 0.05 in the fe world, stop and say so.

**Kappa D3 job: started 11-04 12:14** (`diag_k2_nls_simulation.py --steps 2b,3-6,2bs`, one process). Expected end about 17:00; earlier if the K2b gate stops it after the fe world (about 12:55). The finish line follows here.
**Revised 13:14:** the per-pair K2b fits take about 40 minutes each (the first, fe before-after, ended 12:54), so the end moves to **about 20:00**. The gate result comes about 14:30, and the job stops then if it fails. The finish line follows here.
**Revised 16:25:** the job is still running. The fe-world dose fit (14 parameters, two starts) has been on since 13:46. The machine has been paging since then: 32 GB RAM, 42 GB committed; the job keeps 121 MB of its 757 MB resident and spends 19% of its CPU time in the kernel. That paging slows every later fit as well. New estimates: ideal-world K2b about 19:00–20:00 (the reply follows then); end of the job about **01:00** (range 23:00–03:00), of which the stored-world K2b, last, is about 2 hours. The finish line follows here.
**Revised 16:30 (16:30 amendment):** the stored-world K2b is skipped. A watcher stops the job as soon as `k2_nls_numbers.json` records step 6, so nothing of step 2bs is written. Job end about **23:00** (range 21:00–01:00). The finish line follows here.
**18:54, item 1 done (approved 18:35).**

- `read_menu` in `diagnostics/diag_k2_netflow_simulation.py`:
  - When the binary's row count equals the prep's, it uses the binary, under the same 1e-8 check.
  - Otherwise it takes δ from the closed form and stops if any row is not finite.
  - It returns a record (`delta_source`, `delta_binary_rows`, `prep_rows`, `delta_vs_closed_form_max_abs`) that both K2 scripts write into their numbers JSON. This replaces the bare max-abs value they stored before.
- **LF md5:** `diag_k2_netflow_simulation.py` `eb29157c88d67b4a4ab6415822743bfe`; `diag_k2_nls_simulation.py` `5317999c4c119c75586879d6d3e344d5` (two call-site lines changed).
- **Test on the current production prep, 630,522 rows:**
  - Binary path: δ equals the closed form to 7.1e-15.
  - Closed-form path, forced with a scratch binary one row short: finite on every row, equal to the binary to 7.1e-15, all other columns identical.
- The running job loaded the scripts at 12:14 and is not affected.

**19:16, a snapshot of `52ea6495` for step 7.** I14 will rewrite `processed/market_panel.parquet` and `demand_chars.parquet` in place. K0's `load_panel` reads the panel from that fixed path, so after I14 a build pinned to `52ea6495` would stop at its md5 check. I copied both files, read-only and with their mtimes kept, to `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\904ee55d-75e4-4e82-a3ca-87ddc3397f97\scratchpad\panel_52ea6495\`: `market_panel.parquet` md5 `52ea6495c9195ef38b83bc40fb2014fa`, identical to the source; `demand_chars.parquet` md5 `861a7e243a00197c1cb186704b14c406`, metadata `demand_chars.market_panel_md5` = `52ea6495…`. **I14 can overwrite the originals whenever it is ready.** The side file is pointed at through `DEMAND_CHARS_FILE`. The panel path has no override: Pedro's choice of how to point the build at the copy is in my 19:17 message. The interim-frame sandbox was still empty at 19:15.

**19:37, the 19:30 approvals done.**

- **`read_menu`, keyed on the frame's metadata.**
  - When `demand_chars.market_panel_md5` in the frame starts with `DELTA_BIN_PANEL_MD5 = "d01b8bae"` (a new constant), it uses the binary. The binary must then have the prep's row count and pass the 1e-8 check.
  - Otherwise it takes the closed form, which must be finite on every row. The binary is then not read.
  - The record now carries `frame_panel_md5` and `delta_binary_panel_md5` beside `delta_source`, `delta_binary_rows`, `prep_rows` and `delta_vs_closed_form_max_abs`.
  - **One consequence to note: the production frame (10-29) carries no such key, so it takes the closed form.** No existing frame reaches the binary branch. On that frame the closed form equals the binary to 7.1e-15.
- **`--panel PATH` in `diag_k2_nls_simulation.py`.**
  - `build_ctx(panel)` points `k0.PANEL` at the file. K0's md5 pin and the frame-newer-than-panel time check in `load_demand` then apply to the copy.
  - The numbers record `panel_file` and `panel_md5_read`.
  - When the option is unset, nothing changes.
  - It stops before any read if the file is missing or if an existing `--ctx` file would be read instead.
- **Tests (19:35).** The same production prep in three scratch frames:
  1. Stamped `d01b8bae…`: binary path, binary = closed form to 7.1e-15.
  2. Stamped `52ea6495…`: closed form, finite on all 630,522 rows, equal to the binary to 7.1e-15, other columns identical.
  3. The unstamped production frame: closed form, `frame_panel_md5` None.
  4. On the panel copy, with `K0_PANEL_MD5=52ea6495` and `DEMAND_CHARS_FILE` set, K0's md5 and side-file check passes.
  5. Both `--panel` errors fire before anything is read.
- **LF md5:** `diag_k2_netflow_simulation.py` `55787e721194a12f12e015dd942c2f56`; `diag_k2_nls_simulation.py` `f517111f35b9997065130dd045cfca2b`. These replace the 18:54 pair.
- **Revised end of the job: about 23:30 (22:00–01:00), not 21:00.** Step 3 has 4 of 6 fits; the full-sample fit at the 30% A2 break took an hour, at the 200-iteration cap. Steps 4 to 6 (20 fits) follow.

**Revised 20:43:** the end of the job is about **01:00** (range 00:00–04:00). Step 3's fifth fit (full sample, 60% break) has been running since 19:31; 21 fits follow it. The reply is `KAPPA_REPLY_2026-11-04b.md` (20:43): two decisions (the awake-share definition; where step 7 writes its numbers) and step 7's input checks. The finish line follows here.

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
