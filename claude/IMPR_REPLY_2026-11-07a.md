# IMPR reply 2026-11-07 (a): answers and open items (IMPR, Kappa, tables)

**To:** the main thread, for Pedro. **From:** the IMPR session. **Written** 11-07 22:11; item 1b completed 22:18.

This file supersedes `IMPR_REPLY_2026-11-06a.md` as the live reply; the main thread can retire that file once it has been recorded. The full dated log of 11-05 to 11-07 stays in it until then. The Kappa and tables agents report through IMPR, so there are no separate files.

## 1. Answers to the main thread

**a. Do the BLP standard errors still rest on the approximate Jacobian?** (paper plan S4, asked 11-07.) **No. The red sentence can go.**
- The joint GMM Jacobian is D = [−Z′X, Z′∂δ/∂θ₂]. Its ∂δ/∂θ₂ is the exact IFT Jacobian (`compute_delta_jacobian_gpu` → `exact_delta_jacobian`, `blp_exact_ift.jl`). The log-share Jacobian ∂ln s/∂δ is formed and solved exactly, one quarter at a time, with a quarter's B and D products as one coupled system. It is the same object the gradient uses (`blp_engine_gpu.jl` l.2823–2838; method in the header, l.50–60).
- The only numerical piece is ∂ln s/∂θ₂ at fixed δ\*, by central differences of the shares. That is a derivative of a smooth function, not the old diagonal approximation of ∂ln s/∂δ.
- **Evidence on I2:** FD-CHECK shows FD ≈ IFT at every ext1 optimum. For example E3: 5.17e-6 against 4.84e-6; E1: 3.58e-5 against 3.57e-5.
- **SE method:** WCB is the default (`BLP_SE_METHOD`). ext1's `se_status` is "ok" in all four routines.
- The caveat that remains is the one already in S4: the optimum is stationary, not shown to be global, because ext1's far starts did not move.

**b. The refresh's number check (21:36 item 1). Done 22:12.**
- **Paper numbers (S14, 21:47):** `make_paper_numbers.py` rc 0, **0 MISMATCH**. Statuses: ok 330, pending-_ms983 66, pending-_sc983 12, dropped 1. **The 50 `pending-blp-ext1` keys are released.** 157 entries moved (cluster 150, panel 7). The I2 macros are in Drafts.
- **Compiles (S17, scratch copies, two pdflatex passes):**
  - **V_Friction** (21:45 save): 87 pages, 0 overfull, 0 paper-numbers warnings (22 before the release). Its 15 undefined references all point to labels in the unapplied W5 (b)/W7 blocks (`app:clock`, `app:netflow`, `assum:*`, `sec:disentangle:*`, `tab:tax_clock`). The BLP table's footer spills onto the next page, as it did in the baseline compile.
  - **V_Full:** 102 pages, 0 overfull, 0 undefined references, 46 paper-numbers warnings, all BBL `_ms983`/`_sc983` (BBL is not part of I2).
- **No quoted key is missing, pending or dropped.**

**Sentences the I2 numbers no longer support** (VF = V_Friction 21:45 save; plan = `paper_edits_2026-11-04.tex` 21:59 save):

| where | wording | new value | fixed by |
|---|---|---|---|
| VF 538 | 65+ share "significant at the … percent level" | age65.E3 −0.037, not significant (prints "none") | S2 |
| VF 538 | "Mobile lines carry a small positive coefficient" | mobile E3 +0.0028 → **−0.0008**, E4 −0.0009, neither significant | S2 |
| VF 538, 211 | Selic "Neither is significant" / "not significant" | selic.E4 −1.03\*\*, 95% interval [−8.33, −0.02]; E3 still not significant | S1, S2 |
| VF 540 | Pix "Neither is statistically distinguishable from zero" | pix.E4 −0.90\* (10%); both 95% intervals include zero | S3 |
| VF 585 | Σ "at its zero boundary under (III) and close to it under (IV)" | Σ = 0.000† in both | S5 |
| VF 578 | red sentence "provisional … being re-estimated" | ext1 released; SEs use the exact IFT (item a) | S4 |
| VF 303 | "one of them makes most of the entries" | entrygap groups n 109 → 55 and 89 → 133; rest.banks fourteen → nine | S6 (the path claim holds) |

**Still supported but weaker, with no S-block:**
- VF 538, 182, 211, "local poverty is the strongest correlate": cadunico.E3 1.32 → 2.68\* (10%; the BC 95% interval [−0.14, 7.60] includes zero); E4 3.01\*\*.
- VF 213, "state-owned banks carry a utility premium" (logit): 0.691\*\*\* → 0.295\* (E3), 0.305\* (E4). The `demand.t5.state_owned.E3.sig_level` macro prints 10 now; the sentence's level is the macro's.

**Moved but the sentence still holds** (main thread may want to read the new numbers):
- BLP elasticity −0.83 / −0.87 (VF 182, 213, 583); α −0.216 (0.172), p 0.258, and −0.227 (VF 582: "about three times … five times larger").
- ξ̂ variance shares bank/market/quarter 13/33/12% (VF 587); AR(1) within 0.639 [0.527, 0.748]; own-branches slope 0.424 [0.209, 0.645].
- φ̂ entry 0.917 [0.893, 0.938] (VF 572); the model's 0.984 still above the B interval and inside D's [0.892, 0.9985] (VF 574).
- sleep.meanphi.E3 prints 98 (was 99) at VF 182, 211, 730; 98.4 at VF 589.
- Pix awake share E3 "forty" (41.1); E4 went 17.6 → 44.1 (not quoted).
- dep_anova B: banks 5 → 8%, markets 76 → 75% (VF 295); trunc share E4 46 → 44 (VF 512); share cap 3.6 / 6.3% (VF 901); top share 82 → 81 (VF 486).

**Not quoted, but worth knowing:** Table 4 (IV) `gdp_growth` −0.006 → +0.059\*\*\* and cadunico 0.76 → 3.01\*\*; `demand.phicap.*` B rows at the cap E3 40.3 → 54.2%, E4 39.2 → 50.6% (deposits 14.3 → 23.5%, 13.6 → 17.9%); the national φ̂ mean now runs over 36 quarters rather than 35. The AME table notes now state the reversed draws kept: (III) 21 and 18 of 999, (IV) 21 and 11.

**Quoted keys on a stale vintage** (not rerun, per Pedro's "flag them, refresh later"):
- `sleep.ssel.*` (VF 466, 472) and `\input{tab_state_vector_map.tex}` (VF 999): `DIAG_STATE_SELECTION` of 10-12.
- `tr5.*` (VF 827): R3 (`d01b8bae`). The sentence now sets I2 production (98.4; logit α −0.064) against R3 variants (98.2; −0.075 / −0.067).
- `kappa.k2.*` (VF 710; plan W5 (b) 71, 80; W7 159, 208–253): R3.
- **New today:** `kappa.k0.outside.menu_share.median` (88; VF 710, plan W7 159) is read from `k2_numbers.json`, the K2 world on R3's δ̂. Its K0 siblings are I2.
- **New today:** `phi.entry.gfb.holdout.*` and `gfb.firmtype.*` (VF 574 footnote): `DIAG_PHI_SEPARATION/g_fallback_firmtype/` of 10-29 (R3). The claim "leaves both entry-implied fractions unchanged" now compares R3's fallback (B 0.914) with I2's 0.917. The refresh is one local run: `sleep_ident_entry_dynamics.py --compute-only --g-fallback firmtype --out-dir g_fallback_firmtype`.

**Files** (full key-by-key tables): `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\refresh_i2_1106\key_table.md` (282 quotes, before → after), `…\refresh_i2_1106\final_report.md` (this summary in full), `…\refresh_i2_1106\diff_tables.md` (24 exhibits changed).

**Housekeeping:** C: is 97% full (30 GB free). Two new pickle backups in est3/est4 (`estimation_results.pkl.bak_20261107_211818`, `…_211852`). A temp cleaner removed part of the "before" snapshot at 22:01, after the diffs were computed, so the diffs are complete.

**c. K1 (21:50):** option (i), K1-S, has Pedro's OK (21:51) and is being coded.
- **Method:** (b) + (a). The effects become exp(ψ), solved by unconstrained Gauss–Newton to a tight tolerance and warm-started; the fresh-ALS restart is removed under the flag; ψ is floored at −30, with a count.
- **Scope:** `--smooth-effects`, default off. Only `kappa/k1_netflow_nls.py` changes; `diag_k2_*` are untouched.
- **Added by IMPR:** the exp(ψ) inner problem has a flat direction (w_jm · s_mt scale). One normalization per market is required, or the inner GN step is singular. The agent will report the normalization chosen and the condition number.
- **Acceptance** (all three forms) runs before any pilot, once the refresh ends. Then the three-form re-pilot. The array stays on hold.
- **IMPR's expectation:** smoothing should fix the optimizer, but it will not identify the level, which looked flat (κ̃ 0.21 against 0.63 at the same SSR). With 14–16 clusters the dκ̃ interval may still straddle 0, in which case option (iii) is the paper's statement.
- **23:22, the K1-S acceptance FAILED. The agent stopped before the re-pilot.** No pilot ran, nothing was written to Rout, and the array stays on hold.
  - **Code:** `kappa/k1_netflow_nls.py` `4b9e95d5` → `0fd6714e` (verified by IMPR, CR 0). `--smooth-effects` is off by default, and the flag-off path changes only by the dispatch. `diag_k2_*` and `k1_job.sh` are unchanged.
  - **Check 1, FD against the analytic gradient (savings-only, at the start): FAIL.** For κ̃: −7.64 analytic, −43,690 FD.
  - **Check 2, smoothness: FAIL.** The smooth profile rises above its own start and explodes (7e10 at κ̃ −0.5). The flag-off levels profile still jumps by 0.0018 on κ̃.
  - **Check 3, the zero-residual test 2s with the flag on: FAIL** (error 11.2; it passes with the flag off).
  - **The inner effect problem:** 142 of 300 pilot markets survive. The inner solve hits the 500 cap in all three forms. About 2,000 of 9,102 effects sit at the floor. The conditioning reaches 2.7e9. In 67 of 142 markets, one cell's entry and exit effects are exactly collinear, a second flat direction the per-market normalization does not remove. Three solvers stop at three SSRs (0.22275, 0.22239, 0.22203), so the problem has local optima. A per-market Newton prototype still left 66 markets at its cap.
  - **The agent's diagnosis:** (a) the inner problem is ill-posed on this sample; (b) its normalization shifts only the effects above the floor and then clamps them, which can raise the SSR. That is a code defect, and the likely cause of the test 2s failure.
  - **Options:** (1) fix the normalization (shift every effect of a component, no clamp, with a rule for the collinear entry/exit cells), then rerun checks 1–3 on savings-only; (2) re-specify the effect block, e.g. fewer free exit coefficients (a specification change); (3) option (iii), reporting the sign and a bound on dκ̃ from the existing pilots. The agent recommends (1), then (3) if check 1 still fails.
  - **IMPR's read:** (1) is worth one bounded attempt, because test 2s has zero residuals and should be solvable. A failure there points to the code, not the data. If test 2s passes and check 1 still fails on the pilot sample, the failure belongs to the data, and (3) is the paper's statement. **Pedro chose (1) at 23:27**, with this stop rule. The agent resumed.
  - Scratch: `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\kappa_1105\` (`k1s_accept_so.json`, `k1s_profile.png`, `k1s_cond.json`, `k1s_newton_proto.json`, logs).
- **00:45, the bounded recheck is done. (a) and (b) PASS, (c) FAILS, so by the stop rule option (iii) applies.**
  - **Code:** `kappa/k1_netflow_nls.py` → **`7430007a2ddc13e5584e747645c936ab`** (CR 0, verified by IMPR; only the `--smooth-effects` path changed, default off). `k1_job.sh` `755cdbb4` and `diag_k2_*` are unchanged.
  - **The fix:** the normalization now shifts every effect of a component, floored ones included, with no clamp, so the fit is exactly unchanged. A floored effect is exactly zero.
  - **The collinear cells:** the exit coefficient is capped at c ≤ 100 (the model value is 1). The collinearity is not in the data: it is where the solver escapes, with entry and exit terms growing without bound and cancelling. In 66 cells the exit coefficient reached 1,970–92,863, against a median of 0.83 and a max of 709 in normal cells. A cap does not depend on the solver's path; a collinearity trigger would.
  - **(a) Test 2s, flag on:** 7.0e-13, exact fit. **(b) Tests 1 and 2, flag off:** identical to `4b9e95d5` (5.6e-17 and 3.4e-12). So the code is right.
  - **(c) FD against the analytic gradient (savings-only, at the start): FAIL,** max |FD − analytic|/|grad|∞ = 20.2 (bar 1e-3); κ̃ −13.6 analytic against −2,847 FD. (d) was not run.
  - **The inner solve now converges:** 354 iterations, and a warm re-solve stops after 1 with no SSR fall. SSR/yy is 0.2277; 2,000 effects are floored and 66 exit cells capped.
  - **The agent's reading** (mechanism not tested directly): on these data the inner problem has many nearby local optima and active-set switches. A small parameter move lets the warm solve land on another branch, so the profile jumps even when each solve is exact. It is not differentiable at the scale an FD or LM step sees, which is also why the pilots stall.
  - **For the main thread:** the paper reports the sign and a bound on dκ̃ from the existing pilots (option iii). The repo keeps `--smooth-effects` (default off), and the Bouchet array stays on hold.

**d. BLP stage for CF/BBL: ext1** (main thread). Noted.
- E4's re-fits have been done anyway:
  - ext2 adopted 11-07, Q 0.03585; six starts agree; the freeze came from the logit δ̂ start;
  - extended: two starts agree at Q 0.016938, with SEs refused (K > L), as for E1–E3.
- **The extended fit was adopted at 23:00** (`blp_ext1_adopt.sh` `6fdb4b8e`, `EXT_ALLOW_SE_REFUSED=1`). The three starts agree: Q 0.0169365, 0.0169378 and 0.0169380, with θ₂ within 0.002. e3ext (the lowest Q) is now production. Its σ is 0.384 and α −0.700, against the frozen fit's 0 and −0.227. With K > L these are point estimates only.
- **Brought home 23:45** (archive job 32493892, `CLUSTER_IN\20261107_e4ext`, both sha256 OK):
  - **BLP set:** re-ingested. Only E4 changed: Q path 0.1229 → … → 0.0362 (ext1) → 0.0358 (ext2) → 0.0169 (extended). E1–E3 changed in the timestamp only.
  - **Logs** → `CLUSTER_META\32493892\logs` (265 files).
  - **The multi-start sandboxes' result jsons and logs** (ext1 10-29, ext2 and extended 11-06) → `CLUSTER_META\32493892\multistart` (120 files).
  - **Paper numbers are unaffected:** `make_paper_numbers.py` reads only `BBL_OUTPUT\cluster_processed`, not `BLP_RESULTS`.
  - **Note for the main thread:** `Drafts\Deposit Competition\SUMMARY.md` and `coef_heterogeneity.json` are built on the extended stage (the processor's default), so E4 there now shows α −0.700, not the paper's ext1 −0.227. The paper does not read them.

**e. The four Pix-roster keys (main thread 11-08 00:11). Done 00:21.** `make_paper_numbers.py` rc 0, **0 MISMATCH**, all four keys ok. The values match your checks.

| key | value | prints |
|---|---|---|
| `pix.roster.B.n_mandated` | 16 (of 67 B conglomerates) | sixteen |
| `pix.roster.B.sav_total_brl_bn.2020Q4` | 1,021.853 (63 B conglomerates) | 1,022 |
| `pix.roster.B.sav_nonmandated_brl_mn.2020Q4` | 2.935 (Bank of China Brasil, C0080295; every other non-mandated B conglomerate holds 0) | 2.9 |
| `pix.roster.B.sav_offpix_brl_mn.max` | 2.935 | 2.9 |
| `pix.roster.B.sav_offpix_brl_mn.max_quarter` (companion text key, `\pn{}`) | 2020Q4 | 2020Q4 |

- After 2020Q4 the off-Pix sum falls quarter by quarter, to 2.189 in 2024Q4.
- **Status counts:** ok 330 → 331 (one entry, 5 leaves); otherwise unchanged; 0 changed, 0 removed.
- **Code:** `make_paper_numbers.py` `eaace27c` → **`6ebdef5ba56d0d3ca950cf98d87800b2`** (CR 0, verified by IMPR). The change is a new collector, `collect_pix_roster` (family "panel"). R1 was not rerun.
- **Vintage check by IMPR:** the R1 CSV (11-02 20:06) predates the current panel (`fa38794d`, 11-05). Its B rows are identical to `fa38794d`: all 1,110 B conglomerate-quarters from 2020Q4 to 2024Q4, the same set, with a max savings gap of 0.0 R$ mn. So the keys are on the current panel.

**Question 1, the access date of the roster PDFs: 2026-06-09.**
- The 30 PDFs in `BCB\Egan_et_al_2025_Rep\raw\PIX\roster_cache\` were all written on 2026-06-09 between 18:32 and 18:47, in one run of `scrape_pix_roster.py` (header: "Last edited: 2026-06-09").
- **All 30 came through the Internet Archive**, none directly from the BCB. The script's header says the BCB serves only the current roster and the history lives on the Wayback Machine. Every cached file name starts with its capture timestamp, fetched as `http://web.archive.org/web/<timestamp>id_/<url>`. The captures run from 2020-06-21 to 2026-02-12.
- **Caveat:** no log records the fetch. The date is the files' write time.
- The PDFs are not in the master `DATA_RAW\source_log.csv`; its Pix rows are the S13 IQS PDFs and ChavesPix, fetched 2026-10-27. Adding roster rows is a data-search task if the main thread wants them.

**Question 2, R1b's figure: yes, it is a sum of quarterly stocks.**
- `diagnostics/diag_pix_mandate_2019.py` says "Balances are summed over the four quarters of 2019 and over markets" and labels its output "(summed over quarters)".
- So R$3,225 billion is about 4 × R$806 billion (the average 2019 stock), and R$12 million is about 4 × R$3 million.
- That is consistent with 2020Q4's R$2.94 million of R$1,021.9 billion. The paper's 2020Q4 keys are point-in-time stocks.

**f. The big-5 branch-presence keys (main thread 11-08 00:35).** Sent to the tables agent at 00:43 as T15. The values, the denominator, the five conglomerate codes and the count of MCAs with no B branch will be posted here. Pedro may shut the PC down around 00:55, so if T15 is cut off, the next IMPR session finishes it.

**g. The stale firm-type fallback (`phi.entry.gfb.*`, VF 574 footnote).** Pedro's go at 00:36. It runs on Bouchet as one `sleep_job.sh` entry job: `--g-fallback firmtype --out-dir g_fallback_firmtype`, B = 999, on I2. Its 1.6 MB output folder needs a small download next session.

## 2. State

- **I2 (panel `fa38794d`):**
  - all gates pass; G3 under the documented 2.5% reversed-draw exception (`sleep_ame_twostage.py` `7ada9f87`, `sleep_job.sh` `20a36a56`);
  - fully ingested locally by 21:12;
  - refresh phase 2 done 22:12 (item 1b).
- **Scripts changed 11-06/07 (LF md5):**
  - `make_xi_diagnostics.py` `8092098f` (Panel C layout);
  - `make_paper_numbers.py` `6ebdef5b` (T9, G\*, and T14 `collect_pix_roster` 11-08);
  - `make_k1_table.py` `bdda67be`;
  - `kappa/k1_netflow_nls.py` `0fd6714e` (K1-S `--smooth-effects`, default off; from `4b9e95d5`);
  - `blp_ext1_starts.sh` `75ad5510`, `blp_ext1_start_job.sh` `638e110b`, `blp_ext1_adopt.sh` `6fdb4b8e`;
  - `config/blp_status.toml`: ext1 released.

## 3. Open items

| # | Owner | Item | State |
|---|---|---|---|
| 1 | IMPR | Refresh S17 (compiles) and the number-check table | **done 22:12** (item 1b) |
| 2 | Pedro | S18 equation macros | after Pedro compiles both papers |
| 3 | Kappa | K1-S acceptance FAILED 23:22 (item c). Pedro chose (1) at 23:27: fix the normalization, then one bounded recheck in the order test 2s (flag on), tests 1–2 (flag off), FD check, profile, stopping at the first failure. If test 2s passes but the FD check or the profile fails, option (iii) applies. **00:13: step (a) PASSED** (test 2s with the flag on: max \|est − truth\| 7.0e-13, 15 iterations, 2,294 s), so the 23:22 failure of 2s was the normalization defect. **00:45: (b) passed, (c) FAILED (FD/analytic 20.2), so option (iii) applies** (item c). `k1_netflow_nls.py` `7430007a` | **closed: option (iii), main thread words it** |
| 4 | Pedro | E4 extended adoption (housekeeping): e3ext COMPLETED before 22:57 (elapsed 3 h 51 min). The three starts agree: Q 0.0169365 / 0.0169378 / 0.0169380, θ₂ within 0.002, FD pass, SE refused (K > L). e3ext has the lowest Q. **Adopted 23:00** (Q 0.016936, α −0.700, σ 0.384; 5 files verified; backup `data/output/blp/_superseded_extended_E4_20261107_230006`). Not read by the paper or CF/BBL (ext1) | **done** |
| 5 | main thread | Text check of V_Friction on the I2 macros: S1–S6 cover the broken sentences; cadunico and the logit state-owned premium are weaker with no S-block | table in item 1b |
| 6 | IMPR | Stale vintages: the state-vector diagnostic (`sleep.ssel.*`, VF 466/472/999), TR5 (VF 827), K2 and K0 `menu_share` (VF 710), `g_fallback_firmtype` (VF 574, one local run) | Pedro's go |
| 7 | IMPR | I10: the engine line search and δ start (a δ\* start moves where the logit δ̂ freezes); the ext1 multi-start for global-ness | plan |
| 8 | Pedro | Bradesco 2016 (A plus HSBC) | next rebuild |
| 9 | IMPR | B-registry 11-13 | 11-13 |
| 10 | Pedro | Scratch removal lines (`IMPR_REPLY_2026-11-06a.md`, end). CLUSTER_IN cleanup (11-08 00:05, C: at 98%): tier 1, the redundant copies (~29 GB), in `scratchpad\cleanup_cluster_in_tier1.ps1`; tier 2, R3 and older vintages, rollback only (~18.7 GB), in `…\cleanup_cluster_in_tier2.ps1`. `20261029_tr5` is kept. **Both were run 00:08: C: went from 29 to 73 GB free.** Round 2 (00:31, `…\cleanup_round2.ps1`, 23 targets, 33.9 GB): the interim rebuild sleep runs, the older BLP zips, and the pre-I2 pickle backups | round 2: Pedro to run |
