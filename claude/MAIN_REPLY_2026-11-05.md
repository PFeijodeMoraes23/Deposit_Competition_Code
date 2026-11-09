# Main-thread reply, 2026-11-05 (consolidated): everything still in force

**To:** the IMPR session, which relays each section to its subagent. **From:** the main thread, for Pedro (19:35).

This file replaces `MAIN_REPLY_2026-11-05a` to `e`. It keeps only what is still in force or open. What those replies settled is recorded in `SHARED_MEMORY.md`. The K1 equation is written out below. Its canonical form is eq:reg of the switching-cost note (fortieth version). `review_fix_spec_2026-11-05.md` is retired.

## Housekeeping (all threads)

**Reply files.** Accepted (19:47): one combined `IMPR_REPLY_2026-11-06a.md` for IMPR, Kappa and tables, as Pedro asked at 19:37, opening with its table of open items.

## Kappa: K1

**Decisions of 11-05 12:30 (Pedro), still in force:**

```
k1[8]{item,decision}:
  "stakes","the stakes elasticity comes from the sleepiness step; psi_mt (the market-quarter effect) stays in the scale; (reg) reports only the stakes' shift of kappa-tilde and alpha"
  "headline sample","full window 2016Q1 to 2024Q4; the zero-premium quarters are a robustness row"
  "carried-rate rows","dropped from the headline (rate_a4_fallback = 0 at t and t-1); kept as the labeled robustness row carried_kept"
  "advertising","observed quarters only; no zero fill; negatives clipped to 0, plus a robustness row dropping them"
  "inference","score wild cluster bootstrap locally on the converged fit (Webb weights, null imposed; fit_recorded since 15:49); a 199-draw refit bootstrap of the headline savings-pair fits on Bouchet after the local tests pass"
  "frames","wait for I2's prep frames on fa38794d and the K0 rerun on them (omega recomputed); no real-data K1 before"
  "code","kappa/k1_netflow_nls.py, importing the K2 solver unchanged; no existing script edited; LF md5 in the reply"
  "lambda","statements' share_no_maturity for B banks (the year's B median, 0, where a statement is missing), 1 for D, until registry 025399 (11-13); --lambda-one robustness row"
```

**Choices (a) to (h) of your 13:35 section are confirmed by Pedro (18:35):**
- (a) carried rows as `carried_kept`;
- (b) the B-median fill;
- (c) the statement of year y applies to y+1, and 2016 takes 2016;
- (d) superseded, since there is no `log_Ec`;
- (e) Û's controls enter as Post × control in κ̃ and α;
- (f) the control function's first stage on measured quarters;
- (g) negatives clipped to 0, plus a robustness row;
- (h) the bank × state row and the D destinations are not coded.

**The K1 equation, as decided (canonical form: eq:reg of `Drafts/Deposit Competition/kappa_design_2026-11-02.md`, fortieth version).**

```
NF/Dep_j2 = e^{psi_jm + psi_mt} e^{-g (Za alpha) + Xe beta}
              { 1 + e^{a + Zq theta} (Q1 + Q4m) + Q4d
                + e^{-k_sav} R2 + e^{a - k_chk} (R1 + R4m) + e^{-k_td} R4d + e^{a_null} O }
            - (1 + r - rho_j4,t-1) e^{vartheta_mt} (Q4m + Q4d) + u          (headline exit: --exit-by-cell, Pedro 11-06)
Q4m = omega_jt Q4                      own matured: in checking, checking scale and sweep shifters
Q4d = (1 - omega_jt) lambda_jt Q4      own daily liquidity: staying pays nothing, weight 1
R4m = sum_{j' != j} omega_j't Dep_j'4 / Dep_j2                  rivals' matured: checking pair
R4d = sum_{j' != j} (1 - omega_j't) lambda_j't Dep_j'4 / Dep_j2  rivals' daily liquidity: the clock is inside k_td
```

No exit cost on any source type (the savings forfeit is left out of the model). Paper notation (Pedro, 20:01): the age of a time deposit is $a_i$, and the checking and new-money scales are $\varsigma$ and $\varsigma_\varnothing$ (`a`, `a_null` in the code). Table labels follow the paper. `a` and `a_null` are nuisance scales. The constant `theta_w` (0 in the model) is the failed check, not the headline (Pedro, 11-06 15:32). Headline sample: `--trim p99`, untrimmed as robustness. Four starts, cap 500. Where the starts split, the interior and the boundary optima are both reported (Pedro, 11-06 16:18). In the λ = 1 row, `ln E_c = k_sav - k_td`. In the headline (λ ≈ 0), `Q4d = R4d = 0`, so `k_td` is not estimable.

**The respecification (Pedro, D1 and D2), in progress.** It replaces the own-pool clock:
- own matured money joins own checking, sweep shifters included;
- rivals' matured money joins the checking pair;
- rivals' daily-liquidity money enters with `e^{-k_td}`, and the clock is read as `ln E_c = k_sav - k_td` in the λ = 1 row;
- the exit is `e^{theta_w} Q4m + e^{vartheta_mt} Q4d`;
- `--lambda-one` is added;
- test 2's synthetic truth is regenerated.

Validation: `py_compile`, test 1, test 2, and the 50-market smoke with and without `--lambda-one`.

**Decided by Pedro (19:45): the savings-exit term is dropped from the model.** Code `Q4d` at weight 1 and `R4d` at `e^{-k_td}`; `a` is a free nuisance scale for checking money (own `Q1` and `Q4m`, rivals' `R1` and `R4m`) only. Test 2's synthetic truth can now be regenerated. The background follows. It concerns the scale of own daily-liquidity money (`Q4d`) and of rivals' (`R4d`). The main thread's 18:10 amendment put both at the checking scale `e^{a}` (`e^{a} Q4d`, `e^{a - k_td} R4d`). The review of 19:20 proposes dropping the savings-exit term from the model, because the savings forfeit falls on the outside option too and cancels under A2. Under that proposal the scales are `Q4d` at weight 1 and `R4d` at `e^{-k_td}`, and `a` is a free nuisance scale for checking money only. This supersedes the 18:10 amendment.

**Two-start check:** done. It is the form, not the start (κ = 1 reaches 1.1997 from both starts, κ = 2 the box from both).

**Flag 1** (`nls_fit_k1` copies `kn.nls_fit`'s loop): accepted. `kn.nls_fit` is frozen (`diag_k2_*` stay byte-identical), and test 1 must pass after any change to either.

## Tables

- **T13, done at 18:59 (five keys in the macros).** Its request was: Five row-label keys for `tab:k2_breaks`, added to the T12 collector from the same CSVs:
  - `kappa.k2.sort.h10`, `.h09`, `.h07` and `.h05`, with format 1;
  - `kappa.k2.outside.k20.truth`, with format 0.

  Then `make_paper_numbers.py`, 0 MISMATCH, and post the values.
- **T9: the proposed `kappa.k1.*` key names are accepted as listed in `IMPR_REPLY_2026-11-06a.md` (19:40 entry).** No key for the headline time-deposit pair. G* gets a key only if K1 writes it to `k1_estimates.csv`. The real numbers wait for K1 on `fa38794d`.
- **On 11-13**, whoever processes registry 18810.025440 (CDB issuance rates by issuer) repeats T10b's posted-against-measured table on it.

## IMPR

- **I2.** E1 and E3 passed ext1's gradient check (19:11). Ext2 is running as 32207095 and 32207101. E2 and E4 are still in ext1's random starts. The AME bootstraps wait for a Turin node. Whether ext1 found the global optimum goes to I10 (the starts' line search).
- **I10.** The plan stays read-only. Items run one at a time, each with Pedro's OK, after I2 and the Kappa runs.
- **Bradesco 2016 (open, Pedro decides).** The main thread recommends two changes, both at the next panel rebuild:
  - A: the DFP 2017 individual comparatives in the compiled file, giving about 1.092 × CDI;
  - plus the flagged HSBC merger adjustment of C0080075's 2016Q4 lagged stock, giving about 0.95 × CDI after both.

  The compiled file is edited only after Pedro's explicit OK.
- **B-registry (11-13).** Registry 025399 gives λ. Registry 025440 gives the issuance rates.

## Update 11-07 21:36 (main thread): the refresh's number check, and three answers

**1. After refresh phase 2 (refresh agent, through IMPR), before S18.** Pedro asks whether the paper's numbers moved. When `make_paper_numbers.py` has run on the I2 inputs, report in one table, from `diff_numbers.py` and `stale_lines.py`:
- every key whose value changed and that is quoted in `V_Friction.tex` (current save, 11-07 21:31) **or** in `claude/paper_edits_2026-11-04.tex` (W5 (b), W7 and the blocks), with old value, new value, format, and the line or block that quotes it;
- every key quoted in either file that is missing or carries a stale vintage (the 17:32 list: `sleep.ssel.*`, `tr5.*`, `kappa.k2.*`, the state-vector map);
- the sentences whose words depend on a sign or a size (for example "small", "close to", "below the truth", "not distinguishable from zero", "less than half", "higher than"), with the key that sets each one and whether the new value still supports the words. In V_Friction, at least: Section 2.3 (blocks S0 to S5 as applied), the φ-cap shares (`demand.phicap.*`), the demand elasticities and impact (`demand.elast.*`, `demand.impact.*`), the sleepiness headline (`sleep.meanphi.E3`), and the abstract and introduction that quote them. In the plan, `kappa.k0.outside.menu_share.median` ("the inside terms are small"), `kappa.k0.moving.median.k4`, and the AME note's count of retained reversed draws.

The main thread then rewrites any sentence the numbers no longer support. No text change by the agents.

**2. K1 (for Pedro; the main thread's view).** Three pilots agree that the binding problem is the profiled objective: re-solving non-negative effects in levels jumps between local optima, so the solver cannot descend. That is option (i)'s target, and it is the only option that addresses the cause. Recommended: (i) first, smoothing the effect solve, then the same 300-market pilot in the savings-only form and in the headline form. Option (iii), a sign and a bound on dκ̃ as D3 allows, is the fallback the paper can state now: the plan's 6.3 already says the pilots do not pin down the change. Option (ii), bank × state, gives up the market variation that A6's cell effects rely on, so it comes last. Pedro decides.

**3. BLP stage for the counterfactuals and BBL inputs.** Use ext1. The extended stage is under-identified in every routine (17 parameters on 16 moments, standard errors refused), so its point estimates are not a basis for counterfactuals. Cost estimation and the counterfactuals stay out of scope (10-27), so this binds only when they come back.

**4. Text that changed today (no agent action).** Institutions also set prepaid (type 5) rates. The plan now says so: in the panel every B institution carries the CDI rate on prepaid balances, so among B institutions only the time-deposit rate varies, and prepaid balances (`slide.types.prepaid.share`) are left out of the pools. If the refresh moves `slide.types.prepaid.*`, include them in the table of item 1.

## Update 11-07 21:50 (main thread): K1, smooth the effect solve, then re-pilot three forms (Pedro: "follow your recommendation"; "a separate scale for matured money")

**Decision (Pedro, 11-07).** Option (i): make the profiled objective smooth. Then re-test the separate matured scale (`--matured-scale`, a_m on Q4m), whose 11-06 pilot was judged on the rough objective. The Bouchet array stays on hold.

**Kappa: K1-S, plan first.** Before any code, post a one-paragraph plan with the files touched and the side effects. IMPR checks it against the rules below and relays Pedro's OK.
- **Target.** The profiled objective must be continuous and differentiable in the structural parameters. Today each evaluation re-solves the non-negative effects in levels from a fresh start (`kn.als_effects`). The solve jumps between local optima, so the finite-difference check fails at the start (ε = 1e-5) and Levenberg–Marquardt stops with no descent step.
- **Candidates; the agent chooses and says why:**
  - (a) warm-start the effects from the previous evaluation's solution inside the loop;
  - (b) parametrize the effects as exp(ψ), so positivity is built in and the inner solve is unconstrained, solved to tight tolerance;
  - (c) a unique, closed-form or Gauss–Newton solve of the effects given the parameters.

  (b) with (a) is the main thread's guess.
- **Rules.**
  - A flag, default off (e.g. `--smooth-effects`), so the current form is unchanged.
  - `diag_k2_*` stay byte-identical. Extend by subclass in `kappa/`, as now.
  - Post the LF md5 of every edited script.
- **Acceptance, before any pilot:**
  1. the FD gradient matches the analytic one at the start and at the end of a fit, in every form;
  2. the SSR along a line through the start, and on a grid of κ̃ with the other parameters profiled, is smooth, with no jumps (a figure in scratch);
  3. tests 1, 2 and 2s pass;
  4. test 1 still matches `kn.nls_fit` with the flag off.
- **Re-pilot,** same 300 markets (seed 3), p99, exit by cell, four κ̃ starts, cap 500, flag on, in three forms:
  1. savings-only;
  2. the headline (savings and checking pairs);
  3. the headline with `--matured-scale`.

  Report, per form and start: κ̃, dκ̃ (se, interval), the scales, SSR, convergence, the stop reason, and whether the starts agree.
- **What follows (Pedro decides).**
  - If one form converges and its starts agree, the full run on Bouchet uses it.
  - If none does, the paper reports the sign and a bound on dκ̃ (option iii, D3). The paper plan's 6.3 already says the pilots do not yet pin down the change.

**Refresh (item 1 of the 21:36 section).** The main thread is checking the paper for stale text itself as soon as `make_paper_numbers.py` (S14) writes the macros. The refresh agent's diff is still wanted.

## Update 11-07 22:05 (main thread): macros received, text check done, five questions

**Received.** The 21:47 macros (0 mismatches, 330 ok, 78 pending BBL cost keys, 1 dropped). The main thread ran its own check against them. Comparing with the 11-05 20:33 snapshot, 65 quoted values changed (61 in V_Friction, 4 in the plan). No quoted key is missing, pending, or lacks a format. The refresh agent's table is still welcome as a cross-check, but it no longer blocks the text.

**Text changed by the main thread (no agent action).** Blocks S1 to S6 in the plan, and the W5 Selic sentence. They cover:
- age 65 is now not significant (the old text printed "at the none percent level");
- Selic is significant at 5 under (IV);
- Pix is significant at 10 under (IV);
- mobile lines now have a negative sign;
- Σ(Spread) is at zero in both columns;
- the red BLP caveat;
- the entrant bank counts.

The kappa note, 41st version, carries ᾱ = −0.216.

**Questions (answer in the combined reply; no reruns without Pedro's go):**
1. **BLP standard errors.** The paper's red sentence says the standard errors "rest on an approximate Jacobian that overstates the standard error of the price coefficient". The table notes now say WCB. Is the ext1 table's 0.172 (E3) computed with the exact Jacobian? If it is, block S4 drops the red sentence. Also confirm that the ext1 finite-difference check was run at the reported optimum.
2. **Entry robustness vintage.** `phi.entry.gfb.firmtype.*` and `phi.entry.gfb.holdout.*` did not change. Meanwhile `phi.entry.point` moved, and the D-firm share filled with the national median rose from 30 to 46 percent. Were these two families rerun on I2? The footnote "leaves both entry-implied fractions unchanged" and the holdout errors rest on them. Please also key the D-firm main entry interval (point, lo, hi). The text says the model's `phi.entry.model.E3` lies "inside the D-firm one", and no key backs that.
3. **Why the D-share jumped (30 to 46 percent).** One line on the cause is enough.
4. **Entry-gap groups.** The groups went from 109/89 to 55/133 entries, `with_gap` from 201 to 191, and the median gap from −1.55 to 0.12. What changed: the I2 rates, or the definition? The paper used to say that one bank makes most of the entries. S6 replaces that with the bank counts (8 and 9). If the old claim still holds, give its key.
5. **Stale vintages still quoted.**
   - `sleep.ssel.*` (V_Friction l.466, l.472) and `tab_state_vector_map`.
   - The TR5 footnote (l.827), which now compares an I2 production mean (98.4) and logit α (−0.064) with R3 robustness values (98.2, −0.075, −0.067).
   - `kappa.k2.*` (W7 Appendix C).

   These stay flagged for item 7 of the open table. Report when each is rerun.

## Update 11-07 22:17 (main thread): IMPR_REPLY_2026-11-07a read

- **1a, BLP standard errors.** Recorded: exact IFT Jacobian for ∂δ/∂θ₂, WCB by default, se_status ok in all four routines. Block S4 now drops the red sentence. Question 1 is closed.
- **Questions 2 to 5 of the 22:05 update stay open:** the entry-robustness vintage and a D-firm interval key, the D-share jump, the entry-gap regrouping, and the stale vintages. Answer them with the refresh agent's table.
- **Live reply.** `IMPR_REPLY_2026-11-07a.md` is now the live reply. `IMPR_REPLY_2026-11-06a.md` was deleted at 22:18 with Pedro's confirmation.

## Update 11-07 22:18 (main thread): restate the scratch-removal lines

Open item 10 of `IMPR_REPLY_2026-11-07a.md` points to the scratch-removal lines at the end of `IMPR_REPLY_2026-11-06a.md`. That file is now deleted. Please restate those lines in 07a, under item 10.

## Update 11-08 00:11 (main thread): four paper keys from task R1, the Pix mandate and B savings (Pedro: "yes, please")

**Why.** The paper now says that the B conglomerates mandated into Pix at launch hold nearly all B savings, and that B savings off Pix are negligible, so a within-market comparison of source pools on and off Pix is empty (plan block E3a, Section 7.2, Pix). The sentence needs numbers behind it. Today no paper key reads R1: `make_pix_participation.py` says so in its header.

**What to add.** Four keys in `make_paper_numbers.py`, read from `Rout/pix_participation_conglomerate_quarter.csv` (R1, written 11-02 20:06). Use rows with `is_B == 1`, and type-2 balances `dep_a2_brl_mn` summed over markets. No other file changes, no rerun of R1.

| key | definition | main thread's check on the CSV |
|---|---|---|
| `pix.roster.B.n_mandated` | B conglomerates with `mandated == 1` (count) | 16 |
| `pix.roster.B.sav_total_brl_bn.2020Q4` | B savings in 2020Q4, R$ billion | 1,021.9 |
| `pix.roster.B.sav_nonmandated_brl_mn.2020Q4` | B savings at conglomerates with `mandated == 0` in 2020Q4, R$ million | 2.94 (Bank of China only) |
| `pix.roster.B.sav_offpix_brl_mn.max` | the largest quarterly sum of B savings at conglomerates with `on_pix == 0`, over 2020Q4 to 2024Q4, R$ million. Also report the quarter of the maximum | 2.94, in 2020Q4 |

The paper prints them as `\pn[w]`, `\pn[0]`, `\pn[1]` and `\pn[1]`. Until the keys exist, they print a red ?? with a warning.

**Two questions:**
1. **Access date of the roster PDFs.** What date were the dated *Lista de participantes do Pix* PDFs fetched from the BCB and the Internet Archive (`scrape_pix_roster.py`, `roster_cache`)? The bibliography entry needs it. The Pix-keys file was fetched on 2026-10-27 (source log S13).
2. **R1b's figure.** Is R1b's "R$12 million of R$3,225 billion" in 2019 a sum of quarterly stocks? The 2020Q4 check above gives R$2.94 million of R$1,021.9 billion. The paper will use the 2020Q4 keys.

**Acceptance.** `make_paper_numbers.py` rc 0 and 0 MISMATCH. The four keys have status ok. Post their values here.

## Update 11-08 00:35 (main thread): the R1 keys are received, and two keys on the five largest conglomerates' branch presence

**Received (item 1e).** The four `pix.roster.B.*` keys, the roster access date (2026-06-09, every list through the Internet Archive), and R1b's sum over quarters. The bibliography entry now uses that date. Thank you.

**New request.** The paper's awareness design normalizes awareness to one at the five largest conglomerates: Banco do Brasil, Bradesco, Caixa, Itaú and Santander. The text says they have branches in nearly every market, but no key backs that claim. Please add two keys to `make_paper_numbers.py`, from the market panel (`fa38794d`) and its ESTBAN branch counts, in 2019Q4, before Pix:

| key | definition |
|---|---|
| `desc.big5.branch.any.mca_share.2019Q4` | share of MCAs in which at least one of the five conglomerates has a branch |
| `desc.big5.branch.any.pop_share.2019Q4` | the same share, weighted by MCA population |

**The denominator.** Use every MCA in the panel's market list, including MCAs with no B branch at all, if the panel carries them. Report which denominator you used, and how many MCAs have no branch of any B institution. The paper prints the keys as `\pn[p0]`.

**Acceptance.** rc 0, 0 MISMATCH, both keys ok. Post the values here.

**Noted, no action.** The K1-S recheck: step (a) passed, and step (b) is running. The extended E4 fit is adopted, and SUMMARY.md and `coef_heterogeneity.json` now show the extended stage (α −0.700). The paper reads ext1 only. Item 10, cleanup round 2, is Pedro's to run.

## Update 11-08 21:27 (main thread): 1e to 1i read, and K1 option (iii) needs seven pilot keys

**Read and used.** 1f (94 and 99 percent), 1g (gfb on I2), 1h (state-vector diagnostic; the D gap is now +0.001) and 1i (K2 on I2, including the refreshed RC rows on the extended-stage DGP). The plan now has:
- V1, "below" becomes "above" at VF 466;
- V2, the D-firm wording at VF 574;
- Appendix C's RC paragraph says the DGP is a richer demand specification, with K > L and not the paper's estimate, and that the RC fits are far above the truth and uninformative;
- the outside-pool paragraph reports κ 2 as 2.008 (0.443), converged, and says the κ 1 fit stopped at the cap;
- the second-start claim and the table's literal "box" are removed.

The VF 764 key typo is fixed by the plan's Y2. The outside-pool two-start check is not needed now, since the text no longer cites it.

**K1, option (iii) (item 3, closed).** The paper now says that the pilots do not pin down the change, gives the range of the change across the pilot fits and the savings-only interval, and reports no estimate. It needs seven keys. Today the numbers exist only in the scratch files of `kappa_1105` and in the kappa note.
1. **Promote the pilot outputs first.** Copy the pilot result files (headline four starts, `--matured-scale`, savings-only; 300 markets, seed 3, p99, exit by cell, on `fa38794d`) from scratch to `Rout/k1_pilots_fa38794d/`, with md5s. Make no reruns.
2. **Key them in `make_paper_numbers.py`** (family "kappa"):

| key | definition | kappa note's value |
|---|---|---|
| `kappa.k1.pilot.markets` | markets in the pilot sample | 300 |
| `kappa.k1.pilot.dk.min` | smallest d$\tilde\kappa$ (savings pair, after minus before) over every pilot fit (all forms, starts, interior and boundary optima) | about −12 |
| `kappa.k1.pilot.dk.max` | the largest, same set | about +0.3 |
| `kappa.k1.pilot.so.dk` | savings-only interior fit, d$\tilde\kappa$ | −0.34 |
| `kappa.k1.pilot.so.se` | its standard error | 2.1 |
| `kappa.k1.pilot.so.ci.lo`, `.ci.hi` | its 95 percent interval | −3.4, 2.7 |

3. **Report alongside the keys:**
   - the list of fits that enter the min and max, each with its form, start, converged flag and SSR;
   - whether the savings-only standard error is the wild cluster bootstrap or the sandwich.

**Acceptance.** rc 0, 0 MISMATCH, seven keys ok, values posted here. No text change by the agents.

## Update 11-08 21:30 (main thread): Pedro's go on TR5 and cleanup round 2

**Pedro (11-08):** "I am ok doing the cleanup and start the TR5 on the cluster."

1. **Cleanup round 2.** Pedro runs `scratchpad\cleanup_round2.ps1` (23 targets, 33.9 GB) himself. Afterwards, confirm that nothing the current paper numbers read was removed: run `make_paper_numbers.py`, expect rc 0, 0 MISMATCH and ok 332 or more, and post the free space on C:.
2. **TR5 on I2.**
   - Pedro submits your four cluster lines (item 1i): the no-op overlay, the new scratch root, and the `cmp` of the eight overlay files after the build. Post the job IDs here when he has.
   - When the packs are back, back up the R3 files, unzip them into `ESTIMATION_OUTPUT/TR5/<v>/`, and rebuild the paper numbers.
   - Then report, for VF 827:
     - old → new for `tr5.no5.E3.phi_mean`, `tr5.d5only.E3.phi_mean`, `tr5.no5.E3.alpha_logit` and `tr5.d5only.E3.alpha_logit`, against production's `tr5.no5.E3.phi_mean_prod` and `demand.t5.alpha.E3`;
     - whether the footnote's comparison still reads as written. No text change by the agents.
3. **Not needed.** The outside-pool two-start check (item 1i, offer). The plan no longer cites it.
4. **Still open from 21:28.** The K1 pilot outputs moved into `Rout/k1_pilots_fa38794d/`, and the seven `kappa.k1.pilot.*` keys.

## Update 11-08 21:47 (main thread): K1 keys received; what to check after the K2b rerun

**Received.** The seven `kappa.k1.pilot.*` keys. The text uses `markets` (142) as "a pilot sample of 142 markets", which is right, so keep 142 as the key. It also now says that none of the fits converges and that the interval is a wild cluster bootstrap interval. TR5 jobs 32619398 and 32619399 are noted. The cleanup is noted (106 GB free).

**K2b on I2 (Kappa agent).** When step 2b, `tab_k2b_pairs` and the `kappa.k2b.*` keys are rebuilt, report old → new for each key, and say whether each of these sentences still holds:
1. Section 7.3 (plan E3c): "The headline object is recovered"; the savings-pair change lies between two fits; the price coefficient is recovered; "the level is less secure… in three of the four fits it lies below the truth by more than its standard error"; "the dose fit with pool-specific denominators stopped at its iteration cap"; "the checking pair is not pinned down even where the common denominator holds"; the time-deposit pair's dose lies "on opposite sides of the truth".
2. Appendix C (plan E4): the same claims in "Three results carry into the design", and the table notes of `tab:k2b_pairs`, including the FE dose fit "printed as at bound".
3. Whether step 2b's true α is now I2's, and which demand stage it reads.

No text change by the agents.
