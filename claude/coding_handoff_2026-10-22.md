# Handoff: coding tasks for 2026-10-23 (merged)

**To:** the coding agent (next session). **From:** main thread, for Pedro. **Written:** 2026-10-22, about 20:35.

**Merges and replaces:**
- your self-handoff `coding_handoff_2026-10-23.md` (20:28), deleted after this merge. Its content is kept below;
- the scraping agent's reply `scraping_reply_2026-10-22_type5_rates.md` (about 20:00), deleted. It is now task T5;
- the main thread's earlier coding handoffs of 10-21 and 10-22, deleted at Pedro's request. Everything still open from them is below, and in your memory `resume-2026-10-22-handoff`.

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
decisions[10]{time,item,decision}:
  10-22,Panel rebuild,"before any cluster run; the four pending fixes only"
  10-22,Timing,"R1 on 10-23 morning, machine to itself"
  10-22,Cluster scope,"Sleep + Logit + RC-BLP only (--no-bbl --no-cf); no policy function, BBL or counterfactuals"
  10-22,K4,"after R2, against the new production"
  10-22,Renames,"allowed for code and artefacts; papers and deck stay author-only"
  10-22,Macros (PROPOSED; Pedro to confirm),"one paper_numbers_macros.tex and one writer; generalise --blp-pending to a cluster-pending flag"
  10-22 19:35,Type-5 D1,"a stated or implied zero = nominal 0% balance rate: Stone, C6, Alelo, Cora, PayPal; every other issuer keeps 100% of CDI"
  10-22 19:35,Type-5 D2,"the zeros run from 2020-01 (or the issuer's first type-5 quarter) through 2024Q4"
  10-22 19:35,Type-5 D3,"Banco Original C0080903 type-5 = PicPay's rows, 2020Q2-2024Q3; then C0088022 from 2024Q4"
  10-22 20:21,Type-5 D4 (iti after mid-2023),"evidence first (scraping agent); keep 100% for C0080099 until a follow-up note"
```

**D1 to D3 are the type-5 decision your R2 gate was waiting for.**

## Tasks, in order

```
tasks[7]{id,what,when}:
  R1,"panel rebuild + audits",10-23 morning
  T1,"runners, cluster-pending flag, split of make_followups_1020.py (code only)",while R1 runs
  T5,"type-5 balance rates under D1-D3",after R1 and before the R2 upload
  R2,"cluster Sleep + Logit + RC-BLP, ingest, refresh",after T5
  K4,"RC-BLP without type 5",after R2
  K8,"D entrant-quarters on the national median g",optional (Pedro to confirm)
  K9,"C8(c) sleepiness without type 5",on hold
```

### R1. Panel rebuild

**Command:** `panel_pipeline.py --from 12 --skip 38`.
- It runs steps 12–18, 19–21, 36, 37 and 39, then stage 5. Stages 6 and 7 stay off.
- No step reads `DATA_RAW`.
- It applies the four fixes: the ESTBAN double read (K7), the k calibration, the SOCIEDADES gap, and the 2025+ prepaid rate.

**Baseline:** `claude/rebuild_baseline_2026-10-22/`.
- `firm_class_before.json`;
- `panel_before.json`: 500,186 rows, 131 columns, 536 conglomerates, 3,738 MCAs.
- `rival_mean_spread` exists only per type (`_a1` to `_a5`). It is not missing.

**Audits:**
- `panel_audit_mca_coverage.py --strict`;
- `panel_audit_firm_class.py --compare claude/rebuild_baseline_2026-10-22/firm_class_before.json`;
- shape and totals against `panel_before.json`;
- `panel_demand_chars.py --check`.

**Report:**
- each audit's verdict;
- the 2013–2015 deposit totals, which should halve;
- the type-4 volume without a corrected rate, expected to fall from 2.57% to 0.03%.

Then refresh the panel-only exhibits (T1 runner, or the list under "Exhibit map").

### T1. Code to write during R1 (it does not load the machine)

1. **`exhibits_local.py`:** the panel-only branch, in order, with a staleness check against the panel md5.
   - `make_dep_anova.py`, `make_pix_exposure.py`, `make_state_vector_table.py`;
   - `make_slide_figures.py --only panels,timedep,savings,table --refresh-shares`;
   - `make_slide_figures.py --only band,pricevol,reprice,dpremium,entrygap,relprice,anova,motivtables --png`;
   - plus the local parts of `make_followups_1020.py`.
2. **`exhibits_post_cluster.py`:** the post-run-refresh chain.
   - sleepiness exports (`sleep_ame_twostage.py --attach-from` first), `sleep_desc_clusters.py`;
   - the BLP tables, `make_xi_diagnostics.py`, `sleep_ident_entry_dynamics.py`;
   - `make_paper_numbers.py`, then (after Pedro compiles) `make_paper_equations.py`;
   - the cluster keys of `make_slide_figures.py`.
3. **A cluster-pending flag in `make_paper_numbers.py`,** generalising `--blp-pending`. It marks the cluster-family collectors as pending, so stale estimates print as a red ?? (or as their old value in red with `\PNshowpendingtrue`). **Only if Pedro confirms the macros design.**
4. **Split `make_followups_1020.py` by topic.** Its docstring lists only C1, C3 and C5 and is stale.
   - C1 → `make_phi_variance`, C3 → `make_active_truncation`: both cluster.
   - C2 → `make_entry_carry`: verify its inputs.
   - C5 → `make_dshare_decomp`, C6 → `make_fee_ratio`.
   - C4 merges into `make_pix_exposure`.
   - Update `collect_followups_1020` and the `followups` visual of `make_slide_figures.py`, and keep the quoted keys aliased.

### T5. Type-5 balance rates from posted evidence (D1–D3)

**Basis:**

```
files[3]{file,role}:
  DATA_RAW/issuers/posted_rates/posted_rates.csv,"140 rows; new columns conglomerate (C00... code) and entity_cnpj, filled for the new rows; rate_rule starts with the verdict"
  DATA_RAW/issuers/posted_rates/NOTES.md,"section 'Type-5 follow-up (2026-10-22)': findings by issuer, exact quotes"
  DATA_RAW/issuers/posted_rates/evidence/ + _capture_index.tsv,"every file a row cites, with Wayback timestamps"
```

- **`rate_rule` verdicts:** `ZERO (stated)`, `ZERO to the client implied` (Stone's 20-F), `GAP` (nothing found), `Not the balance` (an opt-in product), `out of window`.
- `valid_from` and `valid_to` are the first and last captures showing a rule. They are not its true start and end dates.

**The finding:**
- No target issuer has a positive rate rule on the balance itself in 2020–2024.
- Where an issuer pays interest, it is on an opt-in product (Stone's Reserva/Guardar, Baú do Ton, Stark's "conta remunerada").
- No act forbids paying on a prepaid balance. Res. BCB 80/2021 art. 22 §8 allows passing the float's yield to the holder. So a zero has to come from the issuer.

**D1. A nominal balance rate of 0% a year** (not "CDI + 0"). The type-5 spread is then the risk-free rate in full, under the panel's own spread convention.

```
zero_rate[5]{conglomerate,issuer,share of type-5 balances,evidence}:
  C0084686,Stone IP,7.21%,"20-F FY2021-23: 'We also collect yield on clients' outstanding balance'"
  C0084844,Banco C6,6.64%,"'O saldo na sua conta C6 nao tem rendimento automatico' (captured 2024)"
  C0087580,Alelo IP,1.97%,"terms: no 'correcao monetaria, juros ou qualquer outro rendimento'"
  C0085317,Cora SCFI,1.93%,"blog 2024-01: 'Ainda nao oferecemos a funcao de conta PJ com rendimento'"
  C0085018,PayPal do Brasil IP,1.22%,"user agreement 2020-2024: no 'juros ou outros rendimentos'"
```

- Every other issuer without a rule keeps the 100%-of-CDI fallback. That includes every GAP row, EBANX (C0085094, whose 2020 EBANX GO line belongs to a different entity) and Stark.

**D2. The zeros run from 2020-01** (or the issuer's first type-5 quarter) through 2024Q4: C6 before 2024-03, Cora before 2024-01, and Stone over 2020–2024.

**D3. Banco Original's type-5 balances are PicPay's.**
- Apply PicPay's balance rows to **C0080903 from 2020Q2 to 2024Q3**, and to **C0088022 (PicPay Bank) from 2024Q4**.
- The panel supports the link:
  - Original's type-5 balance appears in 2020Q2 and reaches R$8.4 bn by 2022Q2;
  - it falls to about R$0.7 bn in 2022Q3, when C0088022 starts reporting about R$8 bn;
  - the remaining R$0.76 bn moves to C0088022 in 2024Q4.
- The link rests on a Moody's report. Check it in IF.data if the conglomerate membership is there.
- PicPay's rows run from 100% of CDI to a promotional 102–210% (2020-12 to 2022), some with balance tiers. Apply each row's headline rate, and list the tiered or promotional rows separately in the diff table so Pedro can confirm them.

**Not in T5 (Pedro has not decided):**
- **Itaú / iti (C0080099, 9.07%) after mid-2023:** keep 100% (D4).
- **The other positive dated rules:** Mercado Pago "até 105%" from 2024-04, Nubank's 30-day rule from 2022-07-25, and bitz's R$10 floor. Keep these issuers at the fallback. Each rule is 100% of CDI or close to it.

**How:**
- **First locate the fallback.** A grep of `panel_ip_rates.py` and `panel_deposit_rates.py` found no type-5 fallback. Find where the type-5 rate is set to the CDI, and report it before changing anything.
- Make the change in that step, reading the rules from a small config or CSV keyed by `conglomerate`, not hard-coded.
- Touch no other deposit type.
- T5 goes after R1 and before the R2 upload. If it needs more than re-running the steps that set type-5 rates, ask Pedro first.

**Acceptance:**
- A diff table by conglomerate × quarter, with the old and new type-5 rate and spread, and the number of rows changed.
- Nothing changes outside type 5 and the seven conglomerates above: the five zeros, plus C0080903 and C0088022.
- The PicPay mapping holds: C0080903 carries PicPay's rows only for 2020Q2–2024Q3, and C0088022 only from 2024Q4.
- The coverage reproduces the scraping agent's shares of 2016–2024 type-5 balances, computed at 20:13 on the current panel. Report them again after the change.

| Status | Share |
|---|---|
| Positive rate, dated rule | 57.0% |
| Zero (D1) | 19.0% |
| Researched, silent (fallback) | 13.4% |
| Not researched, about 170 small issuers (fallback) | 10.6% |

**Report, without acting on it:**
- **Type 5 gains price variation across issuers.** K4 (demand without type 5) and the C8 headline decision change meaning.
- **Paper text goes stale.** Both papers' data appendix and Section 3.2 say that no prepaid rate is observed and that every institution carries the CDI. Once T5 is in production, tell the main thread, which drafts the edits for Pedro.

### R2. Cluster: Sleep + Logit + RC-BLP

**Gate:** T5 is in the panel, or Pedro chose to run R2 on the fallback.

**Steps:**
1. Locally: `cluster_upload.py --stage --stages sleep,blp`. On 10-22 the pre-flight refused only on `demographics_sigma.parquet`, which R1's step 21 refreshes.
2. On the cluster:
   - unzip and check the sha256;
   - resolve the Julia manifest, alone, after any code-bundle upload (`cluster/LAUNCH_SEQUENCE.txt` step 3);
   - `bash pipeline_all.sh --dry-run --no-bbl --no-cf`;
   - `sbatch pipeline_all.sh --no-bbl --no-cf`.
   
   On an empty cluster, the BLP phase builds the sysimages and the draws itself.
3. Download, ingest, then refresh with the post-run-refresh skill.

**Local dry run (10-22):** it stopped at a 90-second limit. The continuation scheduled after Logit carries `--no-bbl --no-cf`, so the flags propagate. This is only partly verified.

**Report:** every paper key whose value changes, old against new, in both papers.

### K4. RC-BLP without type 5 (after R2)

- **Do not upload `upload_type5_20261021.zip`:** it is stale. It sits in `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/CLUSTER_OUT/`.
- Rebuild the variants from the new parquets: `make_demand_type5_variants.py`, then `--logit-compare`, then `--stage-upload`.
- Give Pedro the three handoff commands with the new zip name.
- Build `demand.t6.{no5,d5only}.{E3,E4}.*`. Report the price coefficient, the elasticities and the change in the sample.
- **Flag:** this changes the sample, not the model.

### K8 and K9

- **K8 (optional):** only if Pedro confirms.
- **K9 (on hold):** `sleep_est_e2.py:236` drops only type 3, hard-coded. Do nothing until Pedro decides.

## Decisions waiting for Pedro

1. **T5 timing:** confirm that T5 goes after R1 and before R2, so the cluster run uses the new type-5 spreads. The alternative is R2 on the fallback, with T5 in a later run.
2. **The other positive type-5 rules:** apply Mercado Pago's "até 105%", Nubank's 30-day rule and bitz's R$10 floor, or keep them at 100% of CDI? Also confirm PicPay's headline rates under D3.
3. **D4 (iti after mid-2023):** waits on the scraping agent's evidence.
4. **The macros design** (Decisions table, the PROPOSED row).
5. **Other open items:**
   - the C8 headline sample, after K4;
   - K8;
   - C2: re-verify without Santander;
   - C6, the fee source. The incumbent median is 5.23 → 3.88, against 5.18 → 3.00 in the notes. Account fees now belong to the Fees agent (`claude/fees_handoff_2026-10-22.md`). The tariff-history LAI (18810.025439/2026-13) was denied on 10-22, and Pedro filed the appeal the same day.

## Exhibit map (10-22; details in memory `pipeline-structure-exhibits`)

- **V_Friction**, 27 inputs: the macros file, 15 local exhibits and 11 cluster exhibits.
  - Cluster: the E3/E4 first- and second-stage tables, `est1-4_spec12_stage2_comparison`, `est1-4_spec12_logit_comparison`, `blp_demand_comparison_noseg_spec12`, `tab_xi_diagnostics`, `cluster_imbalance_panelB` and `fig_entry_dynamics_a/b`.
  - The entry figures read the cluster's demand parquet.
- **Deck**, 19 inputs: 10 local and 6 cluster. The inputs of `fig_slide_sleepy_flow` were not checked.
- **None uses the policy function, BBL or counterfactuals.** V_Full still does, on the old panel.
- **The 23 collectors of `make_paper_numbers.py`:**
  - panel (10): `descriptives` (it also reads `cluster_imbalance.json`), `market_structure`, `table1_spreads`, `mca_count`, `slide_deposit_types`, `branch_estban`, `pix_exposure`, `dep_anova`, `slide_motivation`, `state_selection` (inputs not verified);
  - cluster (9): `demand`, `table_rows`, `table4_sidecar`, `type5_robustness`, `xi_diagnostics`, `first_stage_strength`, `entry`, `entry_counts`, `followups_1020`;
  - BBL (4): `bbl`, `polfunc`, `phi_path`, `design`.

## Done on 10-22

- **Bibliography:** `References.bib` 896 → 901 entries.
  - `santander_condicoes_pf_2022` (form 1614-S06, 05/2022);
  - `bradesco_investfacil_2015` and `bradesco_investfacil_2020`;
  - `bcb_circular_3569_2011` and `bcb_circular_3916_2018`. On 10-22 the main thread also updated the 3569 entry's annotation and link to the consolidated text.
  - Biber: 0 warnings and 0 errors on both papers. Pedro has applied C3.
- **R0:** the `raw_paths` switch gives outputs identical to 10-21.
- **Rebuild baseline** recorded.
- **LAI request for BB Rende Fácil** filed: NUP 18882.002114/2026-19.
- **Type-5 rates:** the scraping round is delivered and Pedro has decided D1–D3, giving task T5.

## Flags to keep visible

- **Silence is not zero.** The GAP issuers keep the fallback because nothing says what their balance earns.
- **Santander's default sweep** covers new individual accounts from 05/2022. Whether existing clients were migrated is undocumented. It spans every K3 result.
- **The mandated Pix exposure** varies across only a few large markets.
- **Tesouro Direto values** are registrations, not holdings.
- **`DATA_RAW` is now organised by source.** The sweep evidence is in `DATA_RAW/issuers/sweep_programs/`.
- **Account-fee panel columns** (`listed_fee_*`, stage 7) are mislabelled and repeat the 2026 snapshot. They belong to the Fees agent. Do not fix them under T5.
