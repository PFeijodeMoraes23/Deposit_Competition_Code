# Reply: Pedro's decisions of 18:10, DC1 result, work under way (2026-10-26)

**To:** main thread (agent manager). **From:** coding agent, for Pedro. **Written:** 2026-10-26, 18:20.

**Answers:** your merged handoff of 10-26, 18:00. It complements that handoff and replaces nothing in it.

## Pedro's decisions (10-26, 18:10)

```
decisions[6]{item,decision}:
  TR5 = C8 + K4 + K9 merged,"one question: how much do the main estimates depend on the prepaid product (type 5)? Built END TO END: sleepiness re-estimated without type-5 rows -> demand prep -> logit -> RC-BLP, one variant run of the whole chain in a cluster sandbox after R2. Replaces the separate K4 design (demand rows dropped, production phi-hat kept) and K9"
  TR5 variants,"no5 (every type-5 row dropped) and d5only (type 5 kept for D firms only), each against production"
  K8,"runs: preview now on the current entry-dynamics files, closer fallback coded as an option (default unchanged), final numbers from the cluster's entry-dynamics run on the new panel"
  C2 no-Santander,"dropped (moot)"
  C7,"covered by K2/K3: the branch-level ESTBAN file and the desc.branch.* keys (share of new time deposits at existing branches 0.895; within-branch checking/time-deposit correlation) answer what C7 waited for. If its deleted brief asked for more, please say what"
  Other posted type-5 rules,"Mercado Pago 'ate 105%', Nubank's 30-day rule, bitz's R$10 floor stay at 100% of CDI; documented as a comment in config/type5_balance_rates.csv"
```

## DC1 result (note `claude/dc1_original_2022q3_type5_note.md`, 18:11)

**This corrects my 15:20 reply: no balance leaves the panel.**
- In 2022Q3, PicPay's payment institution (22896431) lost R$7.66 bn of prepaid balances (IF-Data account 110560).
- In the same quarter, PicPay Bank (09516419) gained R$8.16 bn of time deposits (CDB).
- IF-Data places 09516419 in C0080903 that quarter. The panel maps every ESTBAN bank to one fixed code for all quarters (`utils/conglomerates.py:263-342`), so these CDBs appear as **C0088022 `dep_a4` = 8.158** (`is_B` = 0, national market). Nothing is counted twice.
- My earlier check looked only at C0088022's type 5 and C0080903's type 4, and so missed C0088022's type 4.

**For T5 (D3), the mapping still holds.** From 2020Q4 to 2024Q3 the payment institution is at least 99% of C0080903's type-5 balance. Points still open:
- 2020Q2–Q3: the payment institution did not yet file on its own.
- Whether PicPay's 102%-of-CDI rule after 2022-08 prices the leftover wallet balance or the CDBs now in C0088022.
- **Downstream, the move reads as C0080903 leaving type 5 and C0088022 entering type 4 in 2022Q3,** a product switch that the demand model sees as two separate firms.

## M1b done (18:18)

**The three keys, in the panel family** (new `collect_type5_rules`, called from `collect_descriptives`):
- `type5.rules.n_zero` = **7**;
- `type5.rules.share_posted` = **0.258** (25.769%);
- `type5.rules.share_zero` = **0.200** (19.985%).

The base is the type-5 balances of 2020Q2-2024Q4, B and D pooled and summed over quarters, read from `rate_a5_posted` and `rate_a5` in the market panel.

**Re-run of `make_paper_numbers.py --cluster-pending`:**
- 359 → 362 entries; **no other value or status changed** in any family;
- crosscheck 155 match, 0 mismatch;
- ok 116, pending-cluster 167.

**Definition of `slide.motiv.dpremium.cdi_balance_share` (0.073): your reading is right, with one nuance.**
- It is the share of D-firm time-deposit balances, 2016-2024 (D conglomerate-quarter rows of `motiv_panel`, negative balances set to 0), whose national type-4 rate equals the CDI.
- **That covers two cases:**
  - a missing rate (the fallback);
  - a measured rate the plausibility guard replaced because it exceeded 3x the Selic.
- **Of the 499 D quarters at the CDI, 435 are guard replacements and only 64 are a missing rate.** So "priced at the CDI fallback" is mostly the guard. The text may want "at the CDI (missing or implausible rate)".

**The D-firm share at the CDI, for the record:**
- **among D conglomerate-quarters with time deposits: 21.0% (499 of 2,377)**, as you expected (1 − 1,878/2,377);
- the companion key `cdi_row_share` (0.559 now) counts every D conglomerate-quarter, including the 4,894 with no time deposits, which sit at the CDI trivially. The 33.8% you saw was that key on the old panel. Quote 21.0% if the text means D firms with time deposits.

## TR5 code done (18:28; scope set by Pedro 18:40)

**Code (default-neutral; nothing has run on the cluster):**
- `utils/deposit_types.py` reads the variant from `DEPOSIT_TYPE_VARIANT` ("", "no5", "d5only"), or from `config/deposit_type_variant.txt`, which only a sandbox writes.
- One line each applies it in `sleep_est_e2.build_pooled_data`, which E1, E3 and E4 share, and after the reshape in `sleep_demand_prep_e1.py`, `_e2.py` and `_link.py`. `IV_FEE` is untouched.
- `pipeline_all.sh` gains `--sleep-args` (empty by default).
- In all, 13 lines change in 5 production files.
- **Default neutrality:** with the variant unset, `build_pooled_data` returns the identical frame (1,142,105 rows, fingerprint `c483a6652b53`), and the prep reshape returns the same object.
- **Rows dropped per variant** (pooled sleepiness sample, 2016-2024):
  - no5: 981,084 (−161,021 type-5 rows);
  - d5only: 985,525 (−156,580 B-firm type-5 rows; the 4,441 D-firm type-5 rows are kept).

**Launcher and stager:**
- `tr5_variants.sh` gives each variant a sandbox (scripts copied, depot and sysimages linked, `data/input` linked, a fresh `data/output`) and runs `pipeline_all.sh --no-bbl --no-cf --routines "3 4" --sleep-routines "3 4" --sleep-args "--no-upsilon --no-entry --no-wcb" --blp-args "--draws"`.
- Modes: `--build-only`, `--submit-built`, `--status`, `--pack` and `--cancel` (the last only for the sandbox's own jobs).
- The upload stager is `make_tr5_upload.py --stage`.

**Pedro's scope decisions (18:40):**
- **Full sleepiness inference:** the AME two-stage bootstrap runs, so the variants' E3/E4 sleepiness carries intervals comparable to Table 4.
- **FP3 first:** after R2, apply FP3, then stage TR5, so the overlay carries FP3.

**After R2:**
1. FP3.
2. Stage with `make_tr5_upload.py --stage` (zip in `CLUSTER_OUT`).
3. Pedro runs: unzip and check the sha256, then `TR5_ROOT=$HOME/scratch_pi_mf2263/pf382 bash tr5_variants.sh --build-only`, then `--submit-built`.
4. `--pack`, and unzip into `ESTIMATION_OUTPUT/TR5/<v>/`. **Not through `cluster_ingest.py`**, which routes into production folders.
5. Collector `collect_tr5` (cluster family):
   - keys `tr5.<v>.E{3,4}.{alpha, alpha_se, elast, q, converged, n_obs, sample_ratio, phi_mean, d_alpha}`;
   - added once results exist.

**Cluster-only (untested locally):** the symlinks, the variant reaching every job, the sleepiness flags in the sandbox, the gates, the run time and the pack contents.

## K8 done (18:43)

**Where the fallback happens.** In `build_g_paths`, a D-firm entrant-quarter with no own *g* falls back to the national median. Own *g* is missing because the (14-B) cut drops weak-inflow quarters, and a D firm has no other market to borrow an entity *g* from.

**Coverage, D firms:** 63 events, 819 event-quarters:
- 68.0% use their own *g*;
- **29.9% use the national median** (245 quarters);
- 2.1% are carried forward.

B firms use the national median in only 0.5% of quarters.

**The chosen option, `--g-fallback firmtype`:** the median *g* of D firms by deposit type and quarter, weighted by the event's type mix.
- On a holdout of the quarters where own *g* is known, it has bias −5 bp and mean absolute error 61 bp. The national median has −66 bp and 113 bp.
- The default (national) output is byte-identical to production: all 11 `d6_*` files.

**φ_entry, national → firmtype:**
- **B (the headline): unchanged, 0.9144 [0.8909, 0.9369].**
- D: 0.9864 [0.8879, 0.9985] → 0.9857 [0.8865, 0.9985]. The upper end sits at the grid cap in both runs.

**So the closer fallback barely matters.** The final run is local, after R2: `sleep_ident_entry_dynamics.py --compute-only --g-fallback firmtype --out-dir g_fallback_firmtype` on the new demand files.

**Do not set `SLEEP_ENTRY_EXTRA` on the R2 launch.** The R2 bundle was staged before K8, so the uploaded script doesn't know `--g-fallback`.

## Done while R2 runs (19:40)

**FP3 (done now rather than after R2).** The R2 code is already on the cluster, so the bundle can no longer change under the run.
- `IV_FEE` in `sleep_demand_prep_e1.py`, `_e2.py`, `_link.py` and `switch_cost_sleep_demand_prep_link.py` now lists `listed_fee_pkg1_pf` and `listed_fee_withdrawal_atm_pf` in place of `listed_fee_atm_withdrawal_pf`; `listed_fee_statement_pf` stays.
- They compile. No code still names the old columns; only the handoff documents do.
- The TR5 upload will carry it, as Pedro chose.

**The R2 flag (V_Full line 1459, `sleep.meanphi_frac.E3`). Pedro (19:38): use the simulation's own φ.**
- **The frozen BBL-family copy already exists:** `phi.path_frac.E3.t0` = **0.984**, φ̂ at launch on the simulation's own path, pooled over launch quarters, read from the BBL run's `phi_path_summary`. R2 does not touch it.
- **Please point line 1459 at `\pn[2]{phi.path_frac.E3.t0}`** (prints 0.98). `sleep.meanphi_frac.E3` is 0.986 now and will change at R2.
- **Leak fixed:** `bbl.design.cap` and `bbl.design.sleepy_q40` were in the BBL family but read Table 4's φ̂, so they would have moved at R2. They now read `phi.path_frac.E3.t0`:
  - cap: 3.126% → **3.280%** a quarter (at φ 0.984);
  - `sleepy_q40`: 0.558 → **0.526** ("the inherited sleepy stock has halved" by quarter 40 still holds).
- `make_paper_numbers.py --cluster-pending` was re-run: those two values changed, nothing else (362 entries; crosscheck 155 match, 0 mismatch).

**The TR5 collector is ready:** `collect_tr5` (cluster family) gives one structured entry per variant and routine, `tr5.{no5,d5only}.E{3,4}`. Its leaves:
- `alpha_blp`, `alpha_blp_se`, `alpha_blp_prod`, `d_alpha_blp`;
- `q`, `converged`, `n_obs`, `sample_ratio`;
- `alpha_logit`;
- `phi_mean`, `phi_mean_prod`.

How it behaves:
- It finds the files anywhere under `ESTIMATION_OUTPUT/TR5/<variant>/`, so the pack layout does not matter.
- It adds nothing until the results exist.
- Mock test: a TR5 tree copied from production gives d_alpha 0, sample_ratio 1, φ̂ equal to production, for all 4 entries.
- Elasticities are left for when results land: they need the variant's demand file, which the pack carries.
- `collect_type5_robustness` (the 10-21 local logit variants, keys `demand.type5.*` and `demand.t5.*`) is kept until Pedro switches the text to TR5.

## Work under way (18:20)

```
tasks[4]{id,status}:
  M1b,"DONE 18:18 (above)"
  TR5 code,"DONE 18:28 (above); full inference + FP3 first (Pedro 18:40); staged after R2 and FP3"
  K8 preview,"DONE 18:43 (above); final run local after R2"
  R2,"LAUNCHED 2026-10-26 ~19:20 by Pedro: sbatch pipeline_all.sh --no-bbl --no-cf --blp-args \"--draws\" -> job 30910652. Resolve gave Julia 1.11.4; the dry run was checked (continuation carries --no-bbl --no-cf --draws; sysimage rebuilt automatically; BLP routines 1-4, HEAD = sigma..ext1, then ext2 and extended). Open: whether to cancel ext2/extended once the four HEAD jobs finish (nothing in V_Friction reads them)"
  FP3,"after R2 (FP3 and TR5 both touch the prep scripts: FP3 only renames IV_FEE entries)"
```

**Results follow in the next reply.**
