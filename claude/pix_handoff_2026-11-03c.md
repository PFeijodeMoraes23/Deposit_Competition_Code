# Handoff: the Pix agent, 2026-11-03 (one file; replaces every earlier Pix handoff)

## Update 21:30: your 21:10 reply (the K0 pin) accepted

The `K0_PANEL_MD5` override and the age check on the demand frame are accepted as made (Pedro's 20:51 approval); your suggestion that the demand prep record the panel md5 in its parquet metadata is forwarded to the coding thread as part of I10. Nothing else open but the P3 rerun at the new md5.

## Update 22:45: P1 to P3 accepted; nothing open

Your 19:28 reply (`PIX_REPLY_2026-11-03c.md`, removed after processing) and the three notes are accepted. What Pedro decided on your five points, for the record: K1's use-dose first stage uses Mob and G45 jointly (F 65 to 70 under either clustering); the post-launch timing profile uses the quarter-standardized dose `z_payers_pc_q`, not the 2021 ordering; the stakes use the per-relationship balance `bal`, with `hhi4_lag` and `n_banks4_lag` as controls on the instrument; awareness rests on advertising, since branches move too little within a cell. All of it is recorded in `kappa_handoff_2026-11-03.md` for the Kappa agent. Your one open line: rerun `--only p3` when the coding thread announces the new panel md5 in `impr_handoff_2026-11-03.md`. Your scripts `diag_k0_netflow_objects.py` and `diag_p_shifters.py` are the Kappa agent's inputs; do not change them without a summary here. Reply, when there is something, in `PIX_REPLY_2026-11-03d.md`.

## Rules (from the first Pix handoff, now removed)

- V_Friction.tex, the decks and the design note are edited by Pedro and the main thread only.
- Pedro owns git and deletions; give `Remove-Item -LiteralPath` lines.
- Raw files stay untouched.
- Python `C:/venvs/egan/Scripts/python.exe -I`, `MPLBACKEND=Agg`; one heavy process at a time on the machine; read `market_panel.parquet` through pyarrow, once, columns listed in the reply, never through `load_panel_cached`.
- Before editing any existing script, summarize the change and its side effects and wait for Pedro's OK.
- Never guess a schema; read the file or ask. No `\pn` key. Timestamp every update; "wait" means stop.

The K0 specification and its results are in Section 11 of the design note (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`).

**To:** the Pix side session. **From:** main thread, for Pedro. **Written:** 2026-11-03, 21:10.

**What changed since the 19:15 handoff.** K0 is accepted (assessment below, unchanged). The switching-cost design now has its own agent: K2 and K1 moved to the Kappa agent (`kappa_handoff_2026-11-03.md`). Your tasks are the **P** tasks: the shifters and doses that K1 needs, built from the Pix, awareness and market data you already hold. Reply in `claude/PIX_REPLY_2026-11-03c.md`, timestamped. The rules are restated above.

## K0: accepted

```
assess[8]{item,verdict}:
  Run,"accepted: one pyarrow read, columns listed, md5 checked, nothing edited, no key. The stray folder BCB/Egan_et_al_2025_Rep/Rout is empty and Pedro removes it"
  (a) A1,"accepted. The median cell moves 9% of its carried stock, so A1 holds for the linearization. That outflows exceed the awake stock in 86% of outflow cells under the adopted phi-hat is recorded in the design note as a fact about the paper's sleepiness fit at the cell level"
  (a) churn,"the 41% of gross NF that nets within the bank across municipalities set Pedro's decision on K1's left side: bank x market with a booking flag and p99 trim, bank x state as robustness"
  (b),"accepted. 3.3% of log R2 left after both sets of effects: the level of kappa is a cross-sectional comparison; the Pix change is the headline"
  (c),"accepted, and it corrected the note: the vintage is not in rho_j2 but inside rho_j4 through the lumped expense. Coding thread I8 applies the correction; BB agent B5 finds the old-rule stock by bank"
  (d),"accepted, and it is the finding of the day: the access index of M-hat is inverted. Coding thread I1 fixes it before the demand frames are rebuilt"
  (e),"accepted: 11 statement banks; omega applies to the stock carried into y+1"
  (f),"accepted; the six decisions were taken on 11-03, all as the main thread recommended; recorded in kappa_handoff_2026-11-03.md"
```

Also found by the BB agent since: judicial deposits are inside `dep_a4` for every bank (BB 51 to 76 percent of its type-4 stock, Caixa 33 to 71, the state banks 20 to 56) and the rate chain divides CDB expense by a stock that includes them; task I9 removes both. So the panel is being rebuilt (I1, I6, I8, I9); K1 waits for it, and so should anything of yours that reads type-4 stocks or rates.

## Tasks P1 to P3: the shifters K1 needs

All three are market-level or bank-level series, built once, delivered as files in `Rout` with a short note each. None reads type-4 stocks or rates, so none waits for the rebuild. New scripts under `diagnostics/` or `pix/`, your choice, said once.

```
P_tasks[3]{id,what,output}:
  P1,"The Pix-use dose and the exposures at the MCA. U_m: the 2021 mean of Pix payers per inhabitant from the series of Section 7.1.2 (pix_municipal, aggregated to the MCA as the paper does), standardized across B markets; also its quarterly path 2020Q4-2024Q4 (for the timing check of the note's Section 6.6.5). Mob_m: 2019 mobile lines per 100 inhabitants, and the 2019 4G/5G share, at the MCA, standardized, as in tab_pix_exposure_fs. One row per MCA (and per MCA-quarter for the path), with the population weights used","Rout/p1_pix_dose_exposure_mca.csv; Rout/p1_pix_use_path_mca_quarter.csv; a note with the sources and the first-stage F of U on Mob with MCA and with state clusters, on the same sample as 7.1.2"
  P2,"The awareness shifters. adv_jt: advertising expense by conglomerate and quarter 2016Q1-2024Q4 from the Cosif advertising accounts delivered under LAI 18810.023084 (the series the paper already uses in 7.3), in reais and as a share of the conglomerate's deposits; branches_jm,t-1: the lagged count of the conglomerate's branches in the MCA from ESTBAN. One row per conglomerate-quarter and one per conglomerate-MCA-quarter","Rout/p2_advertising_conglomerate_quarter.csv; Rout/p2_branches_conglomerate_mca_quarter.csv; a note on coverage (which conglomerates have the advertising account, which quarters)"
  P3,"The stakes of the note's Section 3.4, eq. (11): S_mt = the balance-weighted standard deviation of time-deposit spreads across the market's banks times the market's mean balance per relationship (from the CCS stock where it exists, else per cell), by MCA-quarter; and its instrument: the dispersion of rivals' cost shifters times the lagged mean balance. Because S uses type-4 spreads, build it on the current panel now and rebuild it when the new md5 lands (one line in your script)","Rout/p3_stakes_mca_quarter.csv (S, S_instrumented, md5 of the panel used)"
```

**Flags.** P1's use series is an equilibrium object; the note instruments it with $\mathrm{Mob}_m\times\mathrm{Post}_t$, so report the first stage. P2's advertising is the excluded shifter of awareness; say whether the account exists for the small B banks, which is where the design needs it. P3 is the only one that touches type-4 spreads; mark the md5.

## Also for you

The AW1 and TED1 specifications and results are now in Section 11 (the record appendix) of the design note (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`, seventeenth version), under their names. Nothing further on them.
