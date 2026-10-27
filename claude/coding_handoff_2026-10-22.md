# Handoff: open coding tasks (merged; updated 2026-10-26 19:10 after your reply of 18:20, extended 18:47)

**To:** the coding agent (next session). **From:** main thread, for Pedro.

**Folds in:**
- your reply `coding_reply_2026-10-26c.md` (18:20, extended 18:47): Pedro's decisions of 18:10 and 18:40, M1b, DC1, TR5 code and K8 done;
- your note `claude/dc1_original_2022q3_type5_note.md` (18:11), kept as the DC1 record;
- your reply of 17:30 (T5a, M1 and FP done, R2 staged) and the earlier merges: your replies and self-handoffs of 10-22 to 10-26, the scraping agent's type-5 reply, and the Fees agent's fee-panel handoff.

## Rules

- **V_Full.tex, V_Friction.tex and the 2026-10-30 deck are edited by the author only.** Deliver data, keys, tables and figures.
- **Renames.** Code, generators, sidecars and exhibit names may be renamed or edited (Pedro, 10-22). A `\pn` key the papers quote keeps its old name as an alias until Pedro switches the paper.
- **Git and files.** Pedro owns git. You cannot delete files: give `Remove-Item -LiteralPath` lines.
- **Raw files stay untouched.** Exhibits are written once into `Drafts/Deposit Competition`.
- **Running Python.** Interpreter `C:/venvs/egan/Scripts/python.exe`, with `MPLBACKEND=Agg`. One heavy process at a time.
- **Data window.** 2016–2024; type-5 balances start in 2020.
- **Cluster commands** are single-line and non-blocking.
- **Habits.** Timestamp every update. "Wait" means stop.

## Decisions

```
decisions[25]{time,item,decision}:
  10-22,Cluster scope,"Sleep + Logit + RC-BLP only (--no-bbl --no-cf); no policy function, BBL or counterfactuals"
  10-22,Renames,"allowed for code and artefacts; papers and deck stay author-only"
  10-22,Macros design,"one paper_numbers_macros.tex and one writer; families panel / cluster / bbl"
  10-22 19:35,Type-5 D1,"stated or implied zero = nominal 0% balance rate; every other issuer keeps 100% of CDI"
  10-22 19:35,Type-5 D2,"the zeros run from 2020-01 (or the first type-5 quarter) through 2024Q4"
  10-22 19:35,Type-5 D3,"Banco Original C0080903 type-5 = PicPay's rows 2020Q2-2024Q3; C0088022 from 2024Q4 (DC1 confirms it from 2020Q4)"
  10-23 11:25,Type-5 D4 FINAL,"Itau/iti C0080099 keeps 100% of CDI throughout: no rule"
  10-23 ~12:00,Type-5 zeros extended,"ShopeePay C0086275 and Ame C0086488 join the D1 zeros"
  10-26,PicPay promotional rules,"keep the headline rates; END the 150% offer on 2021-05-05 (done, T5a)"
  10-26 18:10,Other posted type-5 rules,"Mercado Pago 'ate 105%', Nubank's 30-day rule, bitz's R$10 floor stay at 100% of CDI; documented in config/type5_balance_rates.csv"
  10-26,M1 mode,"--cluster-pending until R2 returns; rerun plain after R2"
  10-26 18:10,TR5 = C8 + K4 + K9,"one question: how much do the main estimates depend on type 5? Built end to end: sleepiness without type-5 rows -> demand prep -> logit -> RC-BLP, one variant run of the whole chain in a cluster sandbox after R2. Replaces the separate K4 design and K9"
  10-26 18:10,TR5 variants,"no5 (every type-5 row dropped) and d5only (type 5 kept for D firms only), each against production"
  10-26 18:40,TR5 inference,"full sleepiness inference: the AME two-stage bootstrap runs, so the variants' E3/E4 sleepiness carries intervals comparable to Table 4"
  10-26 18:40,TR5 order,"after R2: FP3 first, then stage TR5, so the overlay carries FP3"
  10-26 18:10,K8,"runs: preview now; closer fallback coded as an option (default unchanged); final numbers on the new panel after R2"
  10-26 18:10,C2 no-Santander,dropped (moot)
  10-26 18:10,C7,"covered by K2/K3: desc.branch.* keys (n_branches/n_open/n_close by year, share_existing_dtime, corr_dcheck_dtime) answer every item of the C7 brief; closed"
  10-26,FP IV_FEE columns,"listed_fee_pkg1_pf AND listed_fee_withdrawal_atm_pf replace listed_fee_atm_withdrawal_pf; listed_fee_statement_pf stays"
  10-26,FP 2013-2014,"not filled"
  10-26,FP old static columns,dropped
  10-26,FP conglomerate map,"hybrid: banks in ESTBAN keep their static code every quarter (as deposit_panel_codes); every other institution follows the dated IF.data list"
  10-26,FP conglomerate value,"the head institution's price; if the head lists nothing, the median of the members' positive prices"
  10-23,One rebuild,"R1 and T5 in a single rebuild (done 10-26)"
  10-22 20:35,T5 timing,"T5 before R2 (done)"
```

**C7, main-thread answer to your 18:20 question.** The deleted brief asked for nothing beyond what K2/K3 delivered:
- branch-level ESTBAN feasibility;
- `desc.branch.{n_branches,n_open,n_close}.{year}`;
- `desc.branch.share_existing_dtime`;
- `desc.branch.corr_dcheck_dtime`.

All exist in the macros. Its one flag was that quarter-end stocks hide flows within a quarter. C7 is closed.

## Done

```
done[12]{id,result}:
  R1 + T5,"one rebuild 10-26 14:54-15:11; panel md5 4abf639a; K7, k calibration, SOCIEDADES fix; 17 type-5 rules on 9 conglomerates"
  R1 command,"panel_pipeline.py --from 2 --skip 5,6,7,8,9,11,38 --workers 1 (--from takes a STAGE, not a step)"
  T5a,"PicPay 150% rule ends 2021-05-05; only C0080903 2021Q2 changes; type-5 spread (vw) 2.210 -> 2.215 pp; panel md5 0c47f30a"
  Step-16 fix,"panel_deposit_rates.py drops every column it merges, and any _x/_y copy, before the merge"
  M1,"--cluster-pending (17:13): crosscheck 155 match, 0 mismatch; panel family 19 changed values; bbl 122 unchanged"
  M1b,"18:18: collect_type5_rules: type5.rules.n_zero = 7, share_posted = 0.258, share_zero = 0.200; 359 -> 362 entries, nothing else moved"
  cdi_balance_share,"confirmed: D-firm time-deposit balances 2016-2024 whose national type-4 rate equals the CDI; of 499 D quarters at the CDI, 435 are guard replacements (> 3x Selic), 64 a missing rate; 21.0% of D quarters with time deposits"
  DC1,"no balance leaves the panel: 2022Q3 R$7.66 bn left PicPay's wallet (C0080903 type 5) and R$8.16 bn of CDBs appeared at PicPay Bank (C0088022 dep_a4); nothing counted twice; D3 holds from 2020Q4"
  TR5 code,"18:28: utils/deposit_types.py (DEPOSIT_TYPE_VARIANT or config/deposit_type_variant.txt); one line each in sleep_est_e2.build_pooled_data and the three demand preps; pipeline_all.sh --sleep-args; 13 lines in 5 production files; default output identical (1,142,105 rows, fingerprint c483a6652b53); tr5_variants.sh and make_tr5_upload.py"
  K8 preview,"18:43: D entrant-quarters on the national median g 29.9% (245 of 819), B 0.5%; --g-fallback firmtype (D median by deposit type and quarter): holdout bias -5 bp, MAE 61 bp vs -66 / 113 bp national; default output byte-identical (11 d6_* files); phi_entry B unchanged 0.9144 [0.8909, 0.9369]; D 0.9864 [0.8879, 0.9985] -> 0.9857 [0.8865, 0.9985], upper end at the grid cap"
  FP1/FP2/FP4/FP5,"stage-7 fee panel with dated listed prices, 13 services, census stickiness; every check passes"
  R2 staged,"processed/CLUSTER_STAGE/20261026/ (code 7e930f2f..., data 7674adf4...); handed to Pedro 17:03"
```

**Main-thread use of M1b and the definition.** Pedro's type-5 text blocks now resolve with no red `??`. The "a third" sentence now reads "missing or implausibly high", following your nuance. A new appendix block states the 3x-Selic guard. Source: `Drafts/Deposit Competition/paper_edits_type5_2026-10-26.tex`, 19 blocks.

## Tasks, in order

```
tasks[4]{id,what,when}:
  R2,"Pedro launches (do NOT set SLEEP_ENTRY_EXTRA); then download, ingest, exhibits_post_cluster.py --run, M1 plain, report every changed key",when the cluster run returns
  FP3,"IV_FEE rename in the four prep scripts",first after R2
  TR5 run,"make_tr5_upload.py --stage; Pedro builds and submits; --pack; unzip to ESTIMATION_OUTPUT/TR5/<v>/; collect_tr5",after FP3
  K8 final,"local: sleep_ident_entry_dynamics.py --compute-only --g-fallback firmtype --out-dir g_fallback_firmtype on the new demand files; footnote keys",after R2
```

### R2. Cluster: Sleep + Logit + RC-BLP (Pedro launches)

- **Launch line** (yours, 17:03): `sbatch pipeline_all.sh --no-bbl --no-cf --blp-args "--draws"`. The draws are rebuilt because their demographic means come from the new demand parquets.
- **Do not set `SLEEP_ENTRY_EXTRA`.** The R2 bundle was staged before K8, so the uploaded script does not know `--g-fallback`.
- **Expect** G1 to report that φ̂ moved against the promoted values (new panel).
- **When it returns:** download, ingest, `exhibits_post_cluster.py --run` (then `--after-compile` once Pedro compiles), and `make_paper_numbers.py` plain.
- **Report:** every key whose value changes, old against new, in both papers and the deck. Group them by family and flag any sign change or any significance crossing 5 or 10 percent.
- **Stale until R2 returns:** the demand parquets on the cluster, the logit and RC-BLP results and the numbers built from them, `upload_type5_20261021.zip` (do not upload it).
- **One number to flag at R2:** V_Full line 1459 explains the cost-estimation simulation with `sleep.meanphi_frac.E3`. After R2 that key reports the new sleepiness fit, not the one the cost estimation used. Say whether a `bbl`-family copy of the old value exists, or can be added, so Pedro can choose.

### FP3. IV_FEE rename (first after R2)

- **Where:** `IV_FEE` of `sleep_demand_prep_e1.py`, `sleep_demand_prep_e2.py`, `sleep_demand_prep_link.py` and `switch_cost_sleep_demand_prep_link.py`.
- **Change:** replace `listed_fee_atm_withdrawal_pf` with both `listed_fee_pkg1_pf` and `listed_fee_withdrawal_atm_pf`, and keep `listed_fee_statement_pf`. In `switch_cost_*` the rename is text only.
- **TR5 edited the same preps** (one line after the reshape, `IV_FEE` untouched). Check that the FP3 edit leaves the TR5 line in place.

**Consequences of Pedro's FP decisions (record):**
- **Itaú:** its price comes from the fallback (the head is the holding), but equals Itaú Unibanco's own.
- **BB:** stickiness falls from 3.0–7.5 to 0.6–2.5 years.
- **Membership changes:** 45 institutions change conglomerate over 2016–2024, holding 1.1% of listed-price rows.

**FP flags:**
- **Listed is not charged.** These are ceilings, and the Open Finance sample put charged prices near 0.30 of the ceiling.
- **Not listed is not zero.** D rows are covered at 12–20%. Digital and prepaid issuers that charge nothing are in `DATA_RAW/issuers/posted_rates/account_fees.csv`.

### TR5. How much do the main estimates depend on type 5? (after FP3)

**Scope (Pedro, 18:10 and 18:40):**
- whole chain per variant: sleepiness, then demand prep, logit and RC-BLP;
- `no5` and `d5only`, each against production;
- full sleepiness inference: the AME two-stage bootstrap runs, so the E3/E4 intervals are comparable to Table 4.

**Rows dropped** (pooled sleepiness sample, 2016–2024, production 1,142,105):
- `no5`: 981,084 (−161,021 type-5 rows);
- `d5only`: 985,525 (−156,580 B-firm type-5 rows; the 4,441 D-firm type-5 rows stay).

**Launcher:** `tr5_variants.sh` gives each variant its own sandbox and runs:

`pipeline_all.sh --no-bbl --no-cf --routines "3 4" --sleep-routines "3 4" --sleep-args "--no-upsilon --no-entry --no-wcb" --blp-args "--draws"`

**Steps after FP3:**
1. Stage: `make_tr5_upload.py --stage` (zip in `CLUSTER_OUT`).
2. Pedro runs:
   - unzip and check the sha256;
   - `TR5_ROOT=$HOME/scratch_pi_mf2263/pf382 bash tr5_variants.sh --build-only`;
   - then `--submit-built`.
3. `--pack`, and unzip into `ESTIMATION_OUTPUT/TR5/<v>/`. **Not through `cluster_ingest.py`**, which routes into production folders.
4. Add collector `collect_tr5` (cluster family): `tr5.<v>.E{3,4}.{alpha, alpha_se, elast, q, converged, n_obs, sample_ratio, phi_mean, d_alpha}`.

**Before staging, check against Pedro's 18:40 decision:**
- The sleepiness flags drop the wild cluster bootstrap (`--no-wcb`). Confirm that the AME two-stage bootstrap still runs, and that the variants' intervals are built the same way as Table 4's.
- If Table 4's intervals are wild-cluster, say so before staging. Pedro asked for comparable intervals.

**Report, per variant against production:**
- price coefficient, elasticity and their change (`d_alpha`);
- mean sleepy share with its interval;
- rows and balance share dropped, by firm type;
- convergence.

**Flags:**
- **Type 5 now has price variation** (about 20% of its balances at a zero rate). TR5 asks how much the estimates rest on that product, not on a no-variation fallback.
- **Untested locally:** the symlinks, the variant reaching every job, the sleepiness flags in the sandbox, the gates, the run time and the pack contents. Have `--status` show the variant in each job's log before the long stages.
- **Old keys.** `demand.type5.*` and `demand.t5.no5.*` exist in the macros from the earlier design. No paper or deck quotes them. Say whether `collect_tr5` retires them.

### K8. Entry-path g fallback (final run after R2)

- **Run, local:** `sleep_ident_entry_dynamics.py --compute-only --g-fallback firmtype --out-dir g_fallback_firmtype` on the new demand files. Production stays on the national fallback (the default).
- **Keys for a possible footnote** (cluster family, `phi.entry` block; adjust the names to your convention):
  - `phi.entry.gfb.share_national.{B,D}` (share of entrant-quarters on the national median; 0.005 and 0.299 in the preview);
  - `phi.entry.gfb.firmtype.{B,D}.{point,lo,hi}`;
  - `phi.entry.gfb.holdout.{national,firmtype}.{bias_bp,mae_bp}`.
- **Flag:** the D interval's upper end sits at the grid cap (0.9985) under both fallbacks. Report the cap and whether it binds in the bootstrap.

## Decisions waiting for Pedro

1. **K8 headline.** The paper does not yet say that 29.9% of D entrant-quarters borrow the national median g. Main-thread default: keep the national fallback as the headline, since B is unchanged and D moves by 0.0007. Add a footnote with the firm-type fallback as a robustness check once the keys exist.
2. **From DC1, the PicPay 102% rule after 2022-08.** On C0080903 it now prices a leftover wallet balance of about R$0.5–0.8 bn. Main-thread default: keep it, since the wallet is still a prepaid balance and the rule is the only evidence on its rate.
3. **From DC1, the product switch.** In 2022Q3 the demand model sees C0080903 leave type 5 and C0088022 enter type 4, as two separate firms. Main-thread default: leave it as data. Linking the two codes would be a model change.
4. **From DC1, 2020Q2–Q3:** PicPay's payment institution did not yet file on its own. Its rule then equalled the fallback. Record only.
5. **C6:** fees, with the Fees agent.

## Exhibit map (10-22, with your 10-26 corrections)

- **V_Friction**, 27 inputs: the macros file, 15 local exhibits and 11 cluster exhibits.
  - Cluster: the E3/E4 first- and second-stage tables, `est1-4_spec12_stage2_comparison`, `est1-4_spec12_logit_comparison`, `blp_demand_comparison_noseg_spec12`, `tab_xi_diagnostics`, `cluster_imbalance_panelB` and `fig_entry_dynamics_a/b`.
- **Deck**, 19 inputs: 10 local and 6 cluster.
- **None uses the policy function, BBL or counterfactuals.** V_Full still does, on the pre-R1, pre-T5 panel. Its `bbl`-family keys are not marked pending.
- **Collector families of `make_paper_numbers.py`:**
  - **panel:** `descriptives` (with `collect_type5_rules`), `market_structure`, `table1_spreads`, `mca_count`, `slide_deposit_types`, `branch_estban`, `pix_exposure`, `dep_anova`, `slide_motivation`, and the panel part of the former `state_selection`;
  - **cluster:** `demand`, `table_rows`, `table4_sidecar`, `type5_robustness`, `xi_diagnostics`, `first_stage_strength`, `entry`, `entry_counts`, the split follow-ups, `collect_state_levels`, C2 `make_entry_carry`, and `collect_tr5` once TR5 returns;
  - **BBL:** `bbl`, `polfunc`, `phi_path`, `design`.

## Housekeeping (for Pedro)

- **Your FP subagent's scratch** (`unit_patch_out\`, `iq.pkl`, `wide.pkl`, `stick.pkl`, `account_fee_listed_history_check.csv`): give the `Remove-Item -LiteralPath` lines in your next reply.
- **An empty folder can go:**
  `Remove-Item -LiteralPath "C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\BCB\Egan_et_al_2025_Rep\processed\ESTIMATION_OUTPUT\Rout\followups_cache"`
- **Pre-R1 copies** (old panel, old CDB-rate file, old demand corrections) sit in your scratchpad (`r1_before/`), for a before/after comparison if needed.

## Flags to keep visible

- **Silence is not zero.** The GAP issuers keep the fallback because nothing says what their balance earns.
- **"CDI rate" on D time deposits is mostly the guard.** 435 of 499 D quarters at the CDI are measured rates above 3x the Selic rate, replaced. A change to that guard moves the digital premium and the D type-4 spreads.
- **Santander's default sweep** covers new individual accounts from 05/2022. Whether existing clients were migrated is undocumented. It spans every K3 result.
- **The mandated Pix exposure** varies across only a few large markets.
- **Tesouro Direto values** are registrations, not holdings.
- **Paper text on type 5 is stale** until Pedro applies `paper_edits_type5_2026-10-26.tex`. All 19 blocks can go in now.
- **V_Full's cost section** (policy function, BBL, counterfactuals) runs on the pre-R1, pre-T5 panel.
