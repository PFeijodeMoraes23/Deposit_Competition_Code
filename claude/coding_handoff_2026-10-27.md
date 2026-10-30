# Handoff: coding tasks of 2026-10-27

**To:** the coding agent. **From:** main thread, for Pedro. **Written:** 2026-10-27, 14:00; updated 15:40 after your reply of 15:35.

**Replaces** the merged handoff of 10-22 to 10-26, which Pedro removed. Pedro's decisions of 10-27 are also in `claude/SHARED_MEMORY.md`. Read it first.

## Rules (unchanged)

- **V_Friction.tex and the decks are edited by Pedro only.** Paper-text work now targets V_Friction only. Deliver data, keys, tables and figures.
- **Git and files.** Pedro owns git. You cannot delete files: give `Remove-Item -LiteralPath` lines.
- **Raw files stay untouched.**
- **Running Python.** Interpreter `C:/venvs/egan/Scripts/python.exe`, with `MPLBACKEND=Agg`. One heavy process at a time.
- **Data window:** 2016–2024.
- **Cluster commands** are single-line and non-blocking.
- **Habits.** Timestamp every update. "Wait" means stop.

## Where things stand

- **Update 17:45, after your reply of 17:40.**
  - **Done:** E1 (panel `d01b8bae`), FK, and P1 re-run on the new panel.
  - **Running:** R3, job 31020411.
  - **Main thread:** wrote the V_Friction text for E1 (`Drafts/Deposit Competition/paper_edits_E1_2026-10-27.tex`).
  - **One question for you (needed for the firm-type sentence):** how does `panel_digital_flags.py` classify an institution that was in ESTBAN during 2016–2024 but is absent from the December 2024 cross-section (it left or merged before then)? Does it default to B, take an earlier month, or something else? Answer in one line.
- **Update 15:40, after your reply of 15:35:**
  - **The ESTBAN non-filing fix takes precedence (Pedro, 10-27).** Pedro is fixing it with you, and it comes first.
  - P1 and TD1 are done.
  - FP3 was done on 10-26.
  - The TR5 pre-check is clear.
  - FK can start: the sidecar was written at 15:18, after your 15:13 check.
- **R2 (job 30985673) was stopped by Pedro at 15:45.** The next cluster run is R3, on the panel after the E1 fix.
- **The type-5 paper text is applied** in both papers. It quotes:
  - `type5.rules.*`;
  - `slide.types.prepaid.*`;
  - `desc.t1.spread5.D.mean`;
  - `slide.motiv.dpremium.cdi_balance_share`.

  Keep these names.
- **Fees:** the tariff LAI files equal the public census byte for byte (Fees agent, 10-27), so stage 7 needs no re-run.

## Decisions of 10-27 (Pedro)

```
decisions[8]{item,decision}:
  ESTBAN non-filing,"FIX FIRST (Pedro, 10-27 15:40): it takes precedence over every other task"
  Cost estimation,"out of scope for the near future: no policy function, BBL or counterfactual work"
  Pix keys,"YES: build a conglomerate x quarter Pix-key intensity measure (task P1) for V_Friction 7.1.2"
  Tesouro Direto,"cheap check first (task TD1); nothing reads the registry until Pedro sees it"
  B3 by state,"dropped: three state points, mixed units; national series as context only"
  K8,"national g fallback stays the headline; the firm-type fallback becomes a footnote after R3 (task K8)"
  Fee keys,"YES (14:40): the Fees agent writes a sidecar; you add the collector (task FK)"
  Paper text,V_Friction only
```

## Tasks, in order

```
tasks[10]{id,what,when}:
  E1,"ESTBAN non-filing fix (see the E1 section below), then one rebuild; Pedro is on it with you",FIRST
  P1,"DONE 15:35 (pix_keys_conglomerate_quarter.csv); the 7.1.2 design is not built",done
  FK,"collect_fees: read the Fees agent's fee_paper_numbers.json into make_paper_numbers.py",after E1 (light; or alongside if the machine is free)
  TD1,"DONE 15:23 (note assessed and deleted; summary in SHARED_MEMORY). Open: the 2023-09 MCA total exceeds the investor-level count (25.34M vs 25.11M); find the cause before any use of make_tesouro_mca.py output",PARKED (Pedro 10-27): no further TD work
  PC1,"three PicPay diagnostics (APPROVED by Pedro 15:45)",now, BEFORE the E1 rebuild; report, then Pedro picks the route
  R2,"CANCELLED by Pedro (10-27 15:45): job 30985673 stopped; nothing to ingest",cancelled
  R3,"same chain as R2 (Sleep + Logit + RC-BLP, --no-bbl --no-cf, upload manifest fixed) on the E1-fixed panel; then M1 plain and the changed-key report",after E1
  K8,"final entry run with --g-fallback firmtype on the R3 demand files; footnote keys",after R3
  FP3,"DONE 10-26 19:35 (TR5 filter line kept)",done
  TR5,"stage, Pedro runs both variants, pack, collect_tr5",after R3 (same panel as R3)
```

### E1. ESTBAN non-filing read as zero (FIRST)

Your proposal of 15:35 is the basis: a cell with `AGEN_ESPERADAS > 0` and `AGEN_PROCESSADAS = 0` becomes missing, then takes the nearest filed month within the same quarter. A quarter with no filed month stays missing and is logged. Main-thread points to settle in the same change:

1. **Real closures against non-filing.** For each blank cell, check whether the branch reports again in a later month.
   - The 18 Santander single-branch cells in 2024Q4 may be real closures. Use 2025 months only as a diagnostic, never as fill.
   - If a cell never reports again, it is an exit, not a gap. Do not fill it.
2. **Missing must not read as exit.** Check every reader of a bank's presence in a market: the entry-event screen, lags, `rival_mean_spread`, and the incumbent counts. A quarter left missing must not create an exit and a re-entry.
3. **Report, before and after:**
   - the panel md5;
   - the cells filled, by quarter and bank;
   - C6 2024Q1 and BTG 2019Q4 balances by type;
   - every audit;
   - the D-firm and B-firm entry-event counts.
4. **One rebuild only.** If Pedro takes the PicPay re-map route (PC1), or reclassifies PagBank (below), put it in this same rebuild. Ask before running.
5. **Then R3:** the same cluster chain as R2, on the fixed panel. TR5 and K8 final run on that panel too.

**On R2 (job 30985673):** Pedro stopped it (10-27 15:45). R3 on the fixed panel is the next cluster run. Do not ingest any partial R2 output.

**PagBank (C0084813): Pedro's view (15:45) is that it should be D.** It is a digital bank with national reach and no branch network, so the B flag looks like an artifact of the classification rule. Before the E1 rebuild:
1. Report the rule that sets `is_B`, and why it makes PagBank B while C6 (also in ESTBAN) is D.
2. List every conglomerate in the same situation: `is_B = 1` with one or very few markets, or a digital business model. Pedro decides on a rule, not a one-off.
3. **Check for lost prepaid balances.** B-firm type-5 balances are spread over markets by the local type-1 footprint (data appendix, Step 3). A B firm with type-5 balances but no type-1 footprint, as PagBank appears to be, gets zero weight everywhere and its type-5 balances drop out. Report every conglomerate-quarter where IF-Data type-5 balances are positive but the panel's type 5 is zero, with the R$ amounts.
4. If Pedro confirms D, the reclassification goes into the same E1 rebuild.

### P1. Pix-key intensity by conglomerate and quarter (DONE 15:35)

**Main-thread note on your 7.1.2 proposal:** the within-firm design is the right one. Transaction balances (types 1 and 5) are compared with the same firm's time deposits, with conglomerate × type and market × quarter fixed effects. One caveat: 2021Q1 exposure is post-launch, so it already reflects early adoption. Say so in any write-up. CPF keys as the main measure are fine. Not built until Pedro asks.

### P1 (original spec, kept for the record)

**Input:**
- the scraping agent's S13 delivery: `DATA_RAW/BCB/pix_participants/chaves/` (month-end keys by ISPB × PF/PJ × key type, 2020-11..2026-09);
- the agent-compiled `compiled/` crosswalk;
- `make_pix_keys_conglomerate.py`.

**Rules:**
- **Quarter-end months only,** 2020Q4–2024Q4.
- **PicPay (ISPB 22896431):** C0080903 for 2020-11..2024-09, then C0088022. This matches where the panel holds PicPay's balances. The S13 crosswalk maps it to C0088022 in every month, so override it.
- **Every other ISPB:** the S13 crosswalk.

**Output:** a side file, `pix_keys_conglomerate_quarter.csv` (location to your convention), one row per conglomerate × quarter. Columns:
- `keys_pf`, `keys_pj`, `keys_total`;
- PF keys by key type (CPF, phone, e-mail, random);
- `share_pf_national` (the conglomerate's share of all mapped PF keys);
- `keys_pf_per_bal` (PF keys per R$ million of the conglomerate's type-1 plus type-5 balances in the panel);
- `is_B`.

It is not merged into `market_panel` and no estimator reads it. Propose how it would enter the V_Friction 7.1.2 design (intensity × deposit type × firm type), but do not build that.

**Report:**
- mapped share of all keys and of PF keys, by quarter;
- the 10 largest unmapped ISPBs;
- B against D shares of PF keys over time;
- the top 10 conglomerates in 2024Q4.

**Flags:**
- keys are a stock of registrations, not usage;
- a CPF key sits at one institution only, while random keys can be many per person;
- the crosswalk is time-invariant except for the PicPay override.

**No `\pn` keys** until Pedro asks.

### TD1. Tesouro Direto registry against B3 state points (now, cheap)

- **Aggregate** the output of `make_tesouro_mca.py` to state × date.
- **Compare** with the three B3 state points in `DATA_RAW/B3/` (see `COVERAGE.md`):
  - 2019-12 and 2020-03: unique CPFs with a position;
  - 2023-09: custody accounts.
- **Report, per date:**
  - the Pearson correlation in levels and per adult (IBGE population);
  - the Spearman rank correlation;
  - the TD/B3 ratio by state, with the five largest outliers.
- **National check:** monthly TD investors against B3's national unique CPFs (RI spreadsheet), 2016–2024.
- **Deliver** a one-page note answering: does the TD registry rank states the way B3 does? Nothing reads the registry afterwards.
- **Flag:** TD values are registrations, not holdings.

### FK. Fee keys (the sidecar exists: 10-27 15:18, ready now)

**Values to cross-check** after the collector runs (unrounded in the sidecar):
- **POF:** share_paying 0.277; mean_payer 38.19; mean_family 10.59.
- **Disputes:**
  - big5 0.249 (n 194,895);
  - digital 0.031 (n 23,587);
  - big5.2024 0.122.
- **Listed Package I:** median 2016 11.55; 2024 15.95.

- **Input:** `Open-Finance/BCB/Tarifas/processed/fee_paper_numbers.json`, written by the Fees agent (`fees/build_fee_paper_numbers.py`). It holds ten keys:
  - `fees.pof.acct.{share_paying,mean_payer,mean_family}`;
  - `fees.disputes.acct_share.{big5,digital}` with their `.n`, and `fees.disputes.acct_share.big5.2024`;
  - `fees.listed.pkg1.big5.median.{2016,2024}`.
- **Add** `collect_fees` to `make_paper_numbers.py`. It is a local family, never pending. Copy each definition into `paper_numbers.md`.
- **Run** `make_paper_numbers.py --cluster-pending` while R3 is out, or plain after R3.
- **Report:** the crosscheck, and confirm that no other value moved.
- **Formats Pedro will use:** `[p0]` for shares, `[2]` for R$ amounts, `[0]` for counts.

### PC1. PicPay diagnostics (approved 15:45; run before the E1 rebuild)

1. PicPay's share of all type-5 balances by quarter, 2020Q2–2024Q4. Also C0088022's share of D-firm time deposits from 2022Q3.
2. Whether C0088022's 2022Q3 appearance is among the D-firm entry events of the entry-path test (`fig_entry_dynamics_*`, the entry counts), and what the D median path does without it.
3. C0080903's and C0088022's demand residuals (ξ) around 2022Q3, against their own history.

Context: in 2022Q3, R$7.66 bn left PicPay's wallet (C0080903, type 5), and R$8.16 bn of CDBs appeared at PicPay Bank (C0088022, type 4). See SHARED_MEMORY for the two routes that follow.

### R3 follow-up (when the run returns; R2 was stopped)

- **Steps:** download, ingest, `exhibits_post_cluster.py --run` (then `--after-compile` once Pedro compiles), and `make_paper_numbers.py` plain.
- **Report every key whose value changes,** old against new, for V_Friction and the deck. Group them by family, and flag any sign change or any significance crossing 5 or 10 percent.
- **Expect** G1 to report that φ̂ moved (new panel).

### K8. Footnote keys (after R3)

- **Run, local:** `sleep_ident_entry_dynamics.py --compute-only --g-fallback firmtype --out-dir g_fallback_firmtype` on the new demand files. Production stays on the national fallback.
- **Keys** (cluster family, `phi.entry` block; adjust the names to your convention):
  - `phi.entry.gfb.share_national.{B,D}`;
  - `phi.entry.gfb.firmtype.{B,D}.{point,lo,hi}`;
  - `phi.entry.gfb.holdout.{national,firmtype}.{bias_bp,mae_bp}`.
- **Report** whether the D interval's upper end still sits at the grid cap.

### FP3 (done) and TR5 (after R3)

- **FP3:** in `IV_FEE` of the four prep scripts, replace `listed_fee_atm_withdrawal_pf` with `listed_fee_pkg1_pf` and `listed_fee_withdrawal_atm_pf`, and keep the TR5 filter line.
- **TR5** (Pedro, 10-26), with full sleepiness inference:
  1. `make_tr5_upload.py --stage`;
  2. Pedro runs `tr5_variants.sh --build-only`, then `--submit-built`;
  3. `--pack` into `ESTIMATION_OUTPUT/TR5/<v>/`, not through `cluster_ingest.py`;
  4. `collect_tr5`.
- **Before staging, confirm** that the variants' sleepiness intervals are built like Table 4's, given `--no-wcb`.
- **If Pedro chooses to re-map PicPay (PC1 route 2),** that panel change comes before TR5 staging.

## Housekeeping (for Pedro)

- **Your FP subagent's scratch** (`unit_patch_out\`, `iq.pkl`, `wide.pkl`, `stick.pkl`, `account_fee_listed_history_check.csv`): give the `Remove-Item -LiteralPath` lines.
