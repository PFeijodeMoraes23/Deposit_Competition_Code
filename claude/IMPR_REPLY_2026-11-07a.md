# IMPR reply 2026-11-07 (a): answers and open items (IMPR, Kappa, tables)

**To:** the main thread, for Pedro. **From:** the IMPR session. **Written** 11-07 22:11; item 1b completed 22:18; **updated 11-08 21:20** (the section below, and items 1e–1i).

## Update 11-08 21:50: K1 pilot keys done; table audit (one stale table: K2b)

**K1 option (iii): seven keys done (Kappa agent, 21:42).** `make_paper_numbers.py` rc 0, **0 MISMATCH**, ok 332 → 333; `587a7ed14308a9b9adbc79869709fdaf` (CR 0); collector `collect_k1_pilots`.
- **Promotion:** the pilots were copied (no reruns) to `Rout/k1_pilots_fa38794d/{headline, matured_scale, savings_only}/` (`k1_estimates.csv`, `k1_numbers.json`). Raw md5s equal the originals'.

| key | value | your note |
|---|---|---|
| `kappa.k1.pilot.markets` | **142** | 300: **differs**, see below |
| `kappa.k1.pilot.dk.min` | −11.981 (matured scale, start 0, maxit, a_m at its box) | about −12 |
| `kappa.k1.pilot.dk.max` | +0.316 (matured scale, start 0.5, maxit, interior) | about +0.3 |
| `kappa.k1.pilot.so.dk` | −0.341 (savings-only interior, start 1, "no descent step") | −0.34 |
| `kappa.k1.pilot.so.se` | 2.107 | 2.1 |
| `kappa.k1.pilot.so.ci.lo` / `.ci.hi` | −3.414 / 2.732 | −3.4 / 2.7 |

- **Markets:** 300 markets are **drawn** (seed 3), but **142** remain in every fit after the p99 trim and the informative-rows rule (16,701 rows, G 14). The key is 142, and 300 is kept in `detail.markets_drawn`. The text should not say the fits use 300 markets. If you want the drawn count as the key, it is a one-line change.
- **The SE:** `so.se` is the **cluster sandwich** (CR1 by destination conglomerate, G 14); the wild-cluster-bootstrap SE is 2.031 (in the detail). The **interval is the score wild cluster bootstrap** (Webb weights, symmetric percentile-t, 9,999 draws).
- **The 12 fits in the min/max** (forms headline, matured scale and savings-only, starts 0 to 1.5) are all **unconverged** (maxit or "no descent step"). Their dκ̃ runs from −11.98 to +0.32, with SSR/yy between 0.1896 and 0.2220. The full list, with each fit's stop reason and the parameters at their box, is in the agent's report; IMPR can paste it on request.
- **Family:** "panel", like every other `kappa.*` collector; the module has no "kappa" family.

**Table audit (Pedro asked whether any table needs updating after the macro changes).** Macros update on compile; generated tables do not. So every `\input` and `\includegraphics` of V_Friction and of the plan was checked against what its generator reads.
- **V_Friction's 19 inputs and 6 figures are all current:**
  - the 11-05 panel tables (the panel is unchanged since);
  - the I2 logit (11-06);
  - the I2 refresh (11-07 21:15–21:41);
  - `tab_state_vector_map` (regenerated 11-08 16:46).
  - None of today's changed inputs (`g_fallback_firmtype`, the K2 files, `cluster_processed` E4 extended) is read by any of their generators.
- **The plan's new tables:**
  - `tab_pix_exposure_fs.tex`: current (11-07 refresh).
  - `tab_tax_clock.tex`: current (computed from the tax law, no data).
  - **`tab_k2b_pairs.tex` and the ~16 `kappa.k2b.*` keys the plan quotes are STALE.** They come from step 2b, `Rout/k2b_pix_change.csv` of 11-04 (d01b8bae; its true α is R3's −0.117), which the K2 rerun did not include and the 11-07 refresh did not flag.
    - **The step 2b rerun on I2 is planned but not started.** It needs about 6.5–7 h locally:
      - fe before-after pool, fe before-after without weights, fe dose (3.8 h on R3), then the gate;
      - then the ideal before-after and the ideal dose.
    - It needs no step 2 and no step 2bs. The PC shut down at about 22:00 on 11-08, so it runs next session, when the PC can stay on for 7 h.
    - The plan and the command are in scratchpad `kappa_1108\k2_fa38794d_log.txt`.
    - The main thread's 21:47 sentence checklist (7.3 a–g, Appendix C, the table notes, α and the stage) has been handed to the Kappa agent for that report.
- **V_Full** (not checked tonight): its BBL and polfunc tables read `BBL_OUTPUT`/`BLP_RESULTS` `cluster_processed`, and E4's extended files there changed on 11-07. Re-check when V_Full is next compiled.

## Update 11-08 21:35: TR5 submitted, cleanup confirmed, K1 pilot keys in progress (answer to the main thread's 21:27 and 21:30)

- **TR5 on I2 has been submitted.**
  - Before submitting, all three checks passed:
    - the cluster's `tr5_variants.sh`, `deposit_types.py`, `sleep_demand_prep_link.py` and `pipeline_all.sh` match the local I2 md5s (`0e389c71`, `72800e42`, `1e48b66a`, `d56ba4f1`);
    - both sandboxes built "sandbox OK" with the no-op overlay under `$HOME/scratch_pi_mf2263/pf382/tr5_i2`;
    - the `cmp` of the eight overlay files against production printed `same` for all eight.
  - **Jobs:** `pipeline_all` **32619398 (no5)** and **32619399 (d5only)**, routines 3 and 4, sleep args `--no-upsilon --no-entry --no-wcb`, BLP with the draws rebuilt. Expect a day or more.
  - When the packs are back: back up the R3 `ESTIMATION_OUTPUT/TR5/<v>/`, unzip, rebuild the paper numbers, and report the VF 827 keys against production.
  - **`--status`/`--cancel` need the resolved root.** `tr5_variants.sh` finds its jobs by working directory, and SLURM records the physical `/nfs/roberts/scratch/pi_mf2263/…` path, not the `$HOME/scratch_pi_mf2263` symlink. Use `TR5_ROOT=$(readlink -f $HOME/scratch_pi_mf2263)/pf382/tr5_i2`. Checked 21:40: both drivers PENDING (Priority).
- **Cleanup round 2 is confirmed harmless.**
  - Pedro ran it on 11-08 at 00:34; all 23 targets went.
  - C: has **106 GB** free.
  - Every paper-number build since (13:31, 13:35, 15:55, 16:47, 20:06) gave rc 0, 0 MISMATCH, ok 332, so nothing the paper reads was removed.
- **K1 option (iii):** promoting the pilots to `Rout/k1_pilots_fa38794d/` and the seven `kappa.k1.pilot.*` keys went to the Kappa agent at 21:33. The values and the list of fits will be posted here.
- **The outside two-start check:** dropped, as the main thread asked.

## Update 11-08 21:20: what changed today and what needs text work

All builds: `make_paper_numbers.py` rc 0, **0 MISMATCH**, statuses ok 332, pending-_ms983 66, pending-_sc983 12, dropped 1. The script is now `61e91065` (CR 0).

**Your requests:**
- **R1 Pix-roster keys** (item 1e): done 11-08 00:21.
- **Big-5 branch keys** (item 1f): done 13:35. The paper prints **94** % of 3,207 MCAs and **99** % of population. The panel carries only MCAs where some institution books deposits. One MCA has no B branch (`UNKNOWN_1504752`).

**Stale vintages refreshed on I2 (Pedro's GO):**
- **g. Firm-type fallback** (VF 574 footnote): every claim holds.
  - B 0.917, identical to the main estimate; D 0.978 against 0.980.
  - The printed MAEs are now **45 vs 99** bp (were 66 vs 118).
  - "Unchanged" is exact for B and true to 2 dp for D.
- **h. State-vector diagnostic** (VF 466, 472, 999):
  - **VF 466: the D gap flipped sign**, −0.004 → **+0.001**. φ(S_t) now lies *above* the B-market aggregate, but the macro prints the absolute value, so **"below" must become "above"**.
  - VF 472 holds: effective dimension 2.591 → 2.413 (GDP) and 2.356 (young); condition number 7.3 → 29.9.
  - The table moved slightly; its message is unchanged.
- **i. K2 + K0 `menu_share`** (VF 710/714; plan W5 (b), W7):
  - stay probability 0.11 → **0.14**; menu share 88 → **86**; awake share 1.8 → **2.1** %.
  - **Plan 427:** "by up to 0.52" → **0.10**.
  - **Plan 429:** "at the largest cost it runs to the bound … not an estimate" is **now false**: 2.008 (0.443), converged. The table's literal "box" must become the key.
  - Two cells are now unconverged: sorting κ 0.5 at h 1, and the outside pool at κ 1.
  - **The RC rows, plans 125/423/425** (refreshed, Pedro's choice): RC κ̃ 1.659 (se 51) and 2.359 (se 800), unconverged. **The bias flips to above the truth.** Step 7's DGP is E3's *extended* fit, not ext1. The logit rows stay below the truth.

**Found in Pedro's V_Friction compile at 21:17** (92 pages, 0 overfull):
- **Typo at VF 764:** `\pn{pix.rosted.B.n_mandated}` should be `pix.roster.B.n_mandated`. It is the only paper-numbers warning.
- 37 undefined references, all to labels of the blocks not yet applied: `app:clock`, `app:netflow*`, `assum:*`, `prop:app_*`, `sec:disentangle:*`, `tab:tax_clock`, `tab:pix_exposure_fs`.

**S18:** `paper_equations_macros.tex` regenerated from V_Friction's 21:17 `.aux` (31 equations, 116 labels). V_Full was not recompiled, by Pedro's choice.

**Still open:**
- **TR5 on I2** (VF 827): IMPR's four cluster lines (item i) are waiting for Pedro.
- **The K2 outside-pool two-start check** behind plan 429's "a second start gives the same fits" was not rerun (offered).

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

**f. The big-5 branch-presence keys (main thread 11-08 00:35). Done 11-08 13:35.** `make_paper_numbers.py` rc 0, **0 MISMATCH**, both keys ok. Statuses: ok 331 → 332, otherwise unchanged.

| key | value | prints `\pn[p0]` |
|---|---|---|
| `desc.big5.branch.any.mca_share.2019Q4` | 0.9439 (3,027 of 3,207 MCAs) | 94 |
| `desc.big5.branch.any.pop_share.2019Q4` | 0.9906 | 99 |

- **The five conglomerates** (checked against the panel's names, all `is_B` = 1): Banco do Brasil C0080329, Bradesco C0080075, Caixa C0080738, Itaú C0080099, Santander C0080185.
- **Branches:** ESTBAN own branches (`PANEL_INTERMED/estban_own_branches.csv`, `own_br > 0`) in 2019Q4.
- **Denominator:** every MCA of the market panel in 2019Q4, i.e. its market list without the D firms' NATIONAL row: **3,207 MCAs**.
  - The panel carries only MCAs where some institution books deposits. An MCA with no bank at all is not in the panel, so it is not in the denominator.
  - **MCAs with no branch of any B institution: 1**, `UNKNOWN_1504752`, which is also the one MCA without population. 1504752 looks like an IBGE municipality code that the MCA crosswalk does not map; IMPR has not checked it. Dropping it would leave the share at 0.944.
- **Population weights:** `pop_total` of the MCA (constant within an MCA in 2019Q4; checked by the collector).
- **Code:** the tables agent wrote `collect_big5_branches` at 00:47 and was then cut off by the shutdown. IMPR reviewed it on 11-08 and made one fix: the source string labelled the binary parquet with a CRLF-stripped md5 (714d7c5f) instead of its raw md5. It now reads `fa38794d` and also records it in the detail. `make_paper_numbers.py` → **`826285c17ce2ff5d6a8dd42ca72da489`** (CR 0).

**g. The stale firm-type fallback (`phi.entry.gfb.*`, VF 574 footnote).** Pedro's go at 00:36. It runs on Bouchet as one `sleep_job.sh` entry job: `--g-fallback firmtype --out-dir g_fallback_firmtype`, B = 999, on I2. Its 1.6 MB output folder needs a small download next session. It was not submitted that night; **submitted 11-08 ~14:00 as job 32580706.**
- **Done 11-08 15:55.**
  - Job 32580706 COMPLETED in 58 s. The archive `g_fallback_firmtype_I2.tgz` (sha256 `55cf18ff…`, verified) was installed into `DIAG_PHI_SEPARATION\g_fallback_firmtype`; the R3 files are backed up in scratchpad `gfb_1108\before_R3`.
  - `make_paper_numbers.py` rc 0, 0 MISMATCH.
- **The VF 574 footnote on I2: every claim holds.** Only the two printed numbers move.

  | claim / key | R3 (10-29) | I2 | holds? |
  |---|---|---|---|
  | "leaves both entry-implied fractions unchanged": B (`gfb.firmtype.B`) | 0.914 [0.891, 0.937] | **0.917 [0.893, 0.938]**, identical to the main estimate | yes, exactly |
  | the same, D (`gfb.firmtype.D`) | 0.979 [0.889, 0.999] | **0.978 [0.891, 0.999]**, against the main D 0.980 [0.892, 0.9985] | yes to 2 dp; 0.002 apart at 3 dp |
  | "tracks the observed accrual more closely": D MAE, firm type against national (bp, printed `[0]`) | 66 vs 118 | **45 vs 99** | yes |
  | the same, B MAE (not quoted) | 39 vs 70 | 32 vs 52 | yes |
  | "upper end of the D interval at the grid limit" | 0.9985 | 0.9985 | yes |

  - Not quoted: the D firm-type bias changed sign, −8.5 → +21.5 bp. The D holdout quarters fell from 592 to 449.
  - **For the main thread:** "unchanged" is exact for B. For D it is 0.978 against 0.980, which is the same at the precision the paper prints the fractions. The wording is the main thread's call.
- **No longer stale:** `phi.entry.gfb.*`.

**h. The state-vector diagnostic (`sleep.ssel.*`, `tab_state_vector_map`) on I2.** Started 11-08 13:38, about 3 h in all.
- **Both gates passed:** E2 coefficients to 1.7e-13, mean φ 1.021869; E3 mean φ 0.984379, max |Δθ| 7.6e-4.
- The script saves its outputs after every block. A 15:55 build of the paper numbers therefore already read the six blocks the keys range over (`SSEL_RANGE_BLOCKS`, all done), so the `sleep.ssel` level keys are final.
- **Done 11-08 16:47.**
  - The diagnostic ran all 11 blocks (exit 0, 11,208 s). Its meta now records the production spec `IV_RivalSpread x Tech`; the 10-12 run recorded `IV_HausmanFull x Tech`.
  - `make_state_vector_table.py` rc 0 (`tab_state_vector_map.tex` rewritten in Drafts); `make_paper_numbers.py` rc 0, **0 MISMATCH**, statuses unchanged (ok 332).
  - Backups (the 10-12 CSVs and table) are in scratchpad `ssel_1108\before`.
- **Quoted keys (V_Friction and the plan quote the same six):**

  | key | where | 10-12 | I2 | the sentence |
  |---|---|---|---|---|
  | `sleep.ssel.E3.dgap` `[a3]` | VF 466 footnote | −0.004 | **+0.001** (exact 0.00108; max over quarters 0.0031) | **The sign flipped.** The text says φ(S_t) "lies 0.004 below" the B-market aggregate; on I2 it lies **0.001 above**. The macro prints the absolute value, so "below" must become "above" (or "within 0.001 of"). |
  | `sleep.ssel.effdim.spec12` | VF 472 | 3.077 | **2.591** | "lowers the effective dimension from 2.591 to 2.413 (GDP) and 2.356 (young)": still lowers, so it holds |
  | `sleep.ssel.effdim.gdp` | VF 472 | 2.889 | **2.413** | holds |
  | `sleep.ssel.effdim.young` | VF 472 | 2.731 | **2.356** | holds |
  | `sleep.ssel.cond.spec12` `[1]` | VF 472 | 6.2 | **7.3** | "raises the condition number from 7.3 to 29.9": holds |
  | `sleep.ssel.cond.young` `[1]` | VF 472 | 25.2 | **29.9** | holds |

- **Table `tab_state_vector_map` (VF 999).**
  - Share absorbed by the fixed effects: 65+ 0.996 → 0.995; mobile 0.632 → 0.620; GDP 0.848 → 0.844; 4G/5G 0.972 → 0.976.
  - Block rows, (condition number, effective dimension, max VIF):
    - specification (12): (6.160, 3.077, 6.074) → (7.295, 2.591, 7.303);
    - + GDP: → (9.281, 2.413, 11.217);
    - + young: → (29.916, 2.356, 128.238);
    - + both: → (38.520, 2.225, 187.484);
    - + 4G/5G: → (8.599, 2.702, 9.215).
  - The ordering and the message are unchanged: the second age moment is the one that breaks the design.
- **Level keys, not quoted in either file:**
  - E2 under S5: mean φ 1.015 → 1.022, avg 1.041 → 1.048, dep-weighted 0.984 → 0.993; min-var point 0.957 → 0.986.
  - E3: mean 0.983 → 0.984, avg 0.983 → 0.996, dep-weighted 0.978 → 0.983; E3 mean range over the six blocks [0.967, 0.984].
- **No longer stale:** `sleep.ssel.*` and `tab_state_vector_map`. **Still stale:** `tr5.*` (VF 827), `kappa.k2.*` and `kappa.k0.outside.menu_share.median` (VF 710, plan W5 (b)/W7).

**i. The last stale reruns (Pedro: GO, 11-08 16:51).**
- **K2 on I2 (local):** given to the Kappa agent at 16:52. It plans first, makes no script edits, writes a new fa38794d-tagged output and leaves the 52ea6495 records alone. It then rebuilds the paper numbers and diffs `kappa.k2.*` and `kappa.k0.outside.menu_share`.
- **TR5 on I2 (cluster):**
  - `tr5_variants.sh --build-only` always overlays `upload_tr5/scripts` (the 10-29 bundle: `deposit_types.py`, `sleep_est_e2.py`, the three demand preps including `sleep_demand_prep_link.py`, `pipeline_all.sh`, `blp_engine_cpu.jl`, the extra-starts csv) on the production copy. Production now contains all of that code, and the link prep has had the I1 fix since 11-04, so the old overlay would run TR5 on stale code.
  - IMPR's lines therefore point `TR5_UPLOAD` at a **no-op overlay** (a copy of production's own `deposit_types.py`) and `TR5_ROOT` at a new scratch root, so the 10-29 sandboxes and their `built.ok`/`submitted.jid` stay untouched.
  - A `cmp` after the build checks the eight overlay files against production.
  - The packs go to `tr5_results_<v>.zip` as before. Locally they unzip into `ESTIMATION_OUTPUT/TR5/<v>/` after backing up the R3 ones.
  - **Pedro's lines (not run yet; run each one after the previous checks out):**
    1. `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && md5sum tr5_variants.sh utils/deposit_types.py sleep_demand_prep_link.py pipeline_all.sh` → expect `0e389c71`, `72800e42`, `1e48b66a`, `d56ba4f1`.
    2. `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && mkdir -p $HOME/scratch_pi_mf2263/pf382/tr5_i2_overlay/scripts/utils && cp -p utils/deposit_types.py $HOME/scratch_pi_mf2263/pf382/tr5_i2_overlay/scripts/utils/ && TR5_UPLOAD=$HOME/scratch_pi_mf2263/pf382/tr5_i2_overlay TR5_ROOT=$HOME/scratch_pi_mf2263/pf382/tr5_i2 bash tr5_variants.sh --build-only` → expect `no5: sandbox OK`, `d5only: sandbox OK`.
    3. `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && for f in utils/deposit_types.py sleep_est_e2.py sleep_demand_prep_e1.py sleep_demand_prep_e2.py sleep_demand_prep_link.py pipeline_all.sh blp_engine_cpu.jl config/blp_ext1_extra_starts.csv; do cmp -s $f $HOME/scratch_pi_mf2263/pf382/tr5_i2/tr5/no5/scripts/$f && echo "same $f" || echo "DIFF $f"; done` → expect eight `same`.
    4. `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && TR5_UPLOAD=$HOME/scratch_pi_mf2263/pf382/tr5_i2_overlay TR5_ROOT=$HOME/scratch_pi_mf2263/pf382/tr5_i2 bash tr5_variants.sh --submit-built` → expect two job ids.
    - Status: the same command with `TR5_ROOT=$HOME/scratch_pi_mf2263/pf382/tr5_i2` and `--status`. Cancel: the same with `--cancel`.
- **K2 on I2: done 19:24 (Kappa agent).**
  - **Build:** `make_paper_numbers.py` rc 0, **0 MISMATCH**, statuses unchanged (ok 332). No script edits.
  - **Inputs:** panel fa38794d; I2 frames (metadata checked); the draw keys are still valid (the draws are unchanged since 07-27).
  - **Steps:**
    - Step B, the world: `diag_k2_netflow_simulation.py --steps 2` into `Rout\k2_fa38794d\netflow_sandbox` (via `SLEEP_OUT_ROOT`).
    - Steps 5–7: `diag_k2_nls_simulation.py --out Rout\k2_fa38794d`, with a new context pickle, preceded by the one step-2 fit they warm-start from (fe/split κ 0.5: κ̃ 0.4973, converged).
  - **Promotion:** the three top-level files `make_paper_numbers.py` reads (`Rout\k2_numbers.json`, `k2_sorting.csv`, `k2_outside.csv`) were replaced by the fa38794d ones, after an md5-verified backup to `Rout\k2_d01b8bae_toplevel\`. `Rout\k2_step7_52ea6495\` is untouched.
  - **Records still on d01b8bae** (read by no quoted key): `k2_recovery*.csv`, `k2_nls_recovery.csv`, `k2_nls_numbers.json`, `k2_report.md`.
- **Moved quoted keys (printed form):**
  - `world.stay_prob.median`: **0.11 → 0.14** (VF 710; plan 111, 159, 208).
  - `world.awake_share`: **1.8 → 2.1 %** (plan 71, 208).
  - `kappa.k0.outside.menu_share.median`: **88 → 86** (VF 710; plan W7 159).
  - Sorting:
    - κ 0: h 0.9 0.107, h 0.7 0.376, h 0.5 0.766.
    - κ 0.5: h 1.0 0.497 (**now unconverged**, maxit), h 0.9 0.607, h 0.7 0.877, h 0.5 1.291.
    - **`k05.max_excess` 0.52 → 0.10.**
  - Outside pool:
    - κ 0.5: in 0.497 (se 0.016), out −0.006.
    - κ 1: in **1.200 → 0.947, now unconverged**; out −0.054.
    - κ 2: in **box (10.000) → 2.008 (0.443), converged**; out −0.102.
- **Sentences:**
  - VF 714 and plan 231, "weighs little" / "the inside terms are small": still hold (outside option 86 % of the denominator, stay probability 0.14), but weaker.
  - **Plan 427:** "exceeds κ in every case" and "close to −ln h" hold; **"by up to 0.52" becomes 0.10** (at h 0.9 the excess is 0.0015, se 0.025). The κ 0.5, h 1 cell is unconverged.
  - **Plan 429:**
    - **"At the largest cost it runs to the bound … not an estimate" is now FALSE** (2.008, se 0.443, converged).
    - The table's literal "box" must become the `\pn` key.
    - "A second start gives the same fits at the two largest costs" rests on the d01b8bae two-start check, which was not rerun.
- **`kappa.k2.rc.*` (plan 125, 423, 425): NOT refreshed. Decision for Pedro.**
  - `make_paper_numbers.py` hard-codes `K2_STEP7_DIR = "k2_step7_52ea6495"`. Refreshing needs a one-line change to `"k2_fa38794d"`.
  - **IMPR's caveat:** step 7 is hard-wired to the **extended** stage of `cluster_processed/blp_E3_spec_12.json` (`rc_inputs` refuses any other structure). That stage has 17 parameters on 16 moments and its SEs are refused, while the paper's demand is **ext1**. Steps 2, 5 and 6 do not read θ₂, so only the RC rows are affected.
  - The new step 7 does not converge in either cell: RC κ̃ 1.659 (se 51.3) and 2.359 (se 799.8), the checking scale at its box (−8). The RC bias **changes sign** against the 52ea6495 run (0.388 and 0.072, below the truth). The logit refits stay below the truth (0.455, 0.822).
  - **Pedro chose "refresh as is" (20:00).** `K2_STEP7_DIR = "k2_fa38794d"`, and the comment now names the extended-stage DGP. `make_paper_numbers.py` → **`61e910652661eeda40b53be7a67c9116`** (CR 0). Rebuild rc 0, 0 MISMATCH, ok 332.
  - **`kappa.k2.rc` moved:**
    - k05: rc 0.388 (0.328) → **1.659 (51.279)**, bias −0.112 → **+1.159**, α −0.135 → −0.200.
    - k05 logit 0.467 → 0.455.
    - k10: rc 0.072 (0.137) → **2.359 (799.773)**; logit 0.875 → 0.822.
    - Inflow ratio p10/p50/p90: 0.221/0.662/1.436 → 0.048/0.624/1.895.
    - Both RC fits are unconverged.
  - **For the main thread:** plan 125, 423 and 425 need rewriting. The RC fit is now above the truth, with standard errors that make it uninformative. "Neither fit converges" and the logit rows' "below the truth" still hold. The DGP is E3's extended stage, not the paper's ext1.
- **Still open (offer):** the outside-pool two-start check behind plan 429's "a second start gives the same fits" (`k2_outside_twostart.csv`, d01b8bae) was not rerun on I2. It is local and the Kappa agent knows it; Pedro's go.

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
| 2 | IMPR | S18 equation macros | **done 11-08 21:19 for V_Friction**; V_Full when Pedro recompiles it |
| 3 | Kappa | K1-S acceptance FAILED 23:22 (item c). Pedro chose (1) at 23:27: fix the normalization, then one bounded recheck in the order test 2s (flag on), tests 1–2 (flag off), FD check, profile, stopping at the first failure. If test 2s passes but the FD check or the profile fails, option (iii) applies. **00:13: step (a) PASSED** (test 2s with the flag on: max \|est − truth\| 7.0e-13, 15 iterations, 2,294 s), so the 23:22 failure of 2s was the normalization defect. **00:45: (b) passed, (c) FAILED (FD/analytic 20.2), so option (iii) applies** (item c). `k1_netflow_nls.py` `7430007a` | **closed: option (iii), main thread words it** |
| 4 | Pedro | E4 extended adoption (housekeeping): e3ext COMPLETED before 22:57 (elapsed 3 h 51 min). The three starts agree: Q 0.0169365 / 0.0169378 / 0.0169380, θ₂ within 0.002, FD pass, SE refused (K > L). e3ext has the lowest Q. **Adopted 23:00** (Q 0.016936, α −0.700, σ 0.384; 5 files verified; backup `data/output/blp/_superseded_extended_E4_20261107_230006`). Not read by the paper or CF/BBL (ext1) | **done** |
| 5 | main thread | Text check of V_Friction on the I2 macros: S1–S6 cover the broken sentences; cadunico and the logit state-owned premium are weaker with no S-block | table in item 1b |
| 6 | IMPR | Stale vintages: **done 11-08:** `g_fallback_firmtype` (item g) and the state-vector diagnostic (item h; the VF 466 D-gap sign flipped). **Still stale:** TR5 (VF 827, a cluster sandbox run) and K2 + K0 `menu_share` (VF 710) | TR5 and K2: Pedro's go |
| 7 | IMPR | I10: the engine line search and δ start (a δ\* start moves where the logit δ̂ freezes); the ext1 multi-start for global-ness | plan |
| 8 | Pedro | Bradesco 2016 (A plus HSBC) | next rebuild |
| 9 | IMPR | B-registry 11-13 | 11-13 |
| 10 | Pedro | Scratch removal lines (`IMPR_REPLY_2026-11-06a.md`, end). CLUSTER_IN cleanup (11-08 00:05, C: at 98%): tier 1, the redundant copies (~29 GB), in `scratchpad\cleanup_cluster_in_tier1.ps1`; tier 2, R3 and older vintages, rollback only (~18.7 GB), in `…\cleanup_cluster_in_tier2.ps1`. `20261029_tr5` is kept. **Both were run 00:08: C: went from 29 to 73 GB free.** Round 2 (00:31, `…\cleanup_round2.ps1`, 23 targets, 33.9 GB): the interim rebuild sleep runs, the older BLP zips, and the pre-I2 pickle backups | round 2: Pedro to run |
