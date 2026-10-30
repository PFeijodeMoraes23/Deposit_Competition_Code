# Shared memory for every Claude instance on Pedro's machine

Read this before any task that touches the paper. Newest entries first.

## 2026-10-27, 17:45. E1 rebuild done, R3 running (coding reply of 17:40)

- **Panel `d01b8bae`** (was `0c47f30a`); every audit passes. **R3 = job 31020411** (Sleep + Logit + RC-BLP on `d01b8bae`, calendar lags).
- **E1 changes (Pedro's decisions):**
  - ESTBAN non-filing: gaps are filled from the same quarter (18 cells, including C6 2024Q1 and BTG 2019Q4); closures are exits (79 cells); no donor stays missing (43).
  - B/D rule: no retail-deposit test; judged on the 2024-12 cross-section. PagBank, C6 Consignado and Letsbank become D; Arbi becomes B.
  - Type-5 allocation: total-deposit or equal weights when a firm has no checking footprint.
  - The source tier is decided per firm over 2016–2024.
  - **PicPay is one firm (C0088022) in every quarter** (PC1 route 2).
  - Calendar lags.
- **Paper numbers refreshed with `--cluster-pending`.** 39 panel ids moved, and every V_Friction claim still holds. The `fees.*` keys are live (task FK done).
- **Paper text:** `Drafts/Deposit Competition/paper_edits_E1_2026-10-27.tex`, V_Friction only, 9 blocks:
  - the firm-type rule, ESTBAN non-filing, type-5 allocation, the source tier, PicPay as one firm, calendar lags;
  - the 7.2 fee paragraph, with three new bib entries;
  - the 7.2 footnote and a typo.
- **Entry paths:** the charts and the `entrygap` keys redraw after R3. There are now 67 D entry events: PagBank, plus Rodobens, Randon and Andbank, which Pedro kept although their "entry" is a switch of source at 2016Q1.

## 2026-10-27, 15:40. ESTBAN non-filing fix takes precedence

- **The bug:** where a bank did not file a raw ESTBAN month (`AGEN_PROCESSADAS = 0`, blank balances), the panel reads zero deposits. C6 drops out in 2024Q1 (about R$24 bn) and BTG Pactual in 2019Q4 (about R$9 bn), plus 1–27 single cells per quarter (coding reply of 15:35).
- **Pedro is fixing it now, and it comes before every other task.** The fix fills from the nearest filed month in the same quarter, separates real closures from non-filing, and makes sure a missing quarter is not read as an exit. Then one rebuild and R3.
- **R2 (job 30985673) was stopped by Pedro (15:45).** The next cluster run is R3, on the fixed panel. Until R3 returns, the cluster-family numbers keep their old values in red, in V_Friction and in the 10-30 deck.
- **Other results of the 15:35 reply:**
  - P1 done: `Rout/pix_keys_conglomerate_quarter.csv`. D firms passed B firms in PF Pix keys in 2023Q4 (0.364 to 0.549 of PF keys).
  - FP3 was applied on 10-26.
  - The TR5 intervals stay comparable to Table 4: wild-cluster bands exist only for E1/E2, and the bootstrap stays on.
- **PicPay: checks first (Pedro, 15:45).** PC1 runs before the E1 rebuild. Pedro then picks the route, and a re-map would join that rebuild.
- **PagBank (C0084813) should be D (Pedro, 15:45).**
  - It is a digital bank without a branch network, so its B flag looks like a rule artifact.
  - The coding agent reports the classification rule and similar cases. A reclassification would join the E1 rebuild.
  - It also checks whether B firms with prepaid balances but no checking footprint lose those balances when type 5 is spread across markets.

## 2026-10-27. Pedro's decisions (recorded by the main thread, 14:00)

- **Cost estimation is out of scope for the near future (Pedro, 10-27).** Policy function, BBL and counterfactuals stay on the old panel, and nobody re-runs or audits them.
- **Paper text: V_Friction only.** Edit plans target `V_Friction.tex`. V_Full gets no more text edits unless Pedro asks. The type-5 text is applied in both papers (10-27). It quotes `type5.rules.*`, `slide.types.prepaid.*`, `desc.t1.spread5.D.mean` and `slide.motiv.dpremium.cdi_balance_share`, so keep those key names.
- **The handoffs of 10-21 to 10-26 were removed by Pedro** (coding, scraping, fees). The main thread writes new ones when needed.
- **R2 is running on the cluster** (Sleep + Logit + RC-BLP on the rebuilt panel, `--no-bbl --no-cf`). Until it returns, the cluster-family numbers print their old value in red.
- **Pix keys by participant: YES, build the intensity variable.**
  - Source: the scraping agent's S13 (`DATA_RAW/BCB/pix_participants/`, `make_pix_keys_conglomerate.py`).
  - PicPay's keys (ISPB 22896431) go to C0080903 for 2020-11..2024-09 and to C0088022 after, matching where the panel holds PicPay's balances (type-5 decision D3).
  - Use: V_Friction Section 7.1.2, the convenience channel.
- **Tesouro Direto as a local outside option: cheap check first.** Compare the registry (`make_tesouro_mca.py`, by market and quarter) with B3's state points (2019-12, 2020-03, 2023-09). Nothing reads it until Pedro sees the check.
- **TD1 result (coding agent, 15:23; note deleted after assessment).**
  - **Outputs:** script `diagnostics/diag_tesouro_vs_b3.py`; files `Rout/tesouro_vs_b3_state.csv` and `Rout/tesouro_vs_b3_national.csv`.
  - **Cross-section: the registry ranks states like B3.** Spearman 0.95–0.99; Pearson per 1,000 inhabitants 0.98–0.99, at 2019-12, 2020-03 and 2023-09.
  - **Level and trend: it does not track B3.** It is 4–6 times B3 by state, highest in AP and TO and lowest in DF, RS and RJ.
  - **The registry is cumulative since 2002 and never drops inactive investors.** Its ratio to the active TD investors of B3's RI series rises from 4 (2016) to 10 (2024). Growth correlations are only 0.43–0.64.
  - **Main-thread reading:** usable as a cross-sectional, pre-period measure (for example 2019 registrations per capita by market). Not usable as a time-varying covariate in levels.
  - **A time-varying measure of active investors would need the Tesouro Transparente "Operações do Tesouro Direto" files** (2014–2026, monthly, about 1.1 GiB compressed), joined to the registry's state and city by investor code. Whether that file carries the investor code is to be confirmed.
  - **Open bug:** at 2023-09 the MCA output totals more than the investor-level count (25.34M vs 25.11M), cause unknown. Fix it before any use of `make_tesouro_mca.py` output.
  - **PARKED (Pedro, 10-27).** No further Tesouro Direto work until Section 7 uses an outside option. If it does, build active investors from the operations files rather than using the registry.
- **B3 investors by state: dropped.** Public B3 sources give only three state points with mixed units. The national series stay as context only.
- **Fee keys: YES (Pedro, 14:40).** Ten `fees.*` keys support V_Friction 7.2: fees are a holding cost at incumbents and near zero at digital banks.
  - The Fees agent writes `BCB/Tarifas/processed/fee_paper_numbers.json` (`claude/fees_handoff_2026-10-27.md`).
  - The coding agent adds `collect_fees` (its task FK).
  - No LAI for BCB complaints by type. The complaint tables are not joined to the panel yet. F6 is skipped and F5 stays on hold.
  - **Done 15:25; the fee workstream is closed.** Sidecar values:
    - POF: 27.7% of families pay; R$38.19 a month per paying family (2018 prices);
    - fee disputes over account complaints: big five 0.249 (2016: 0.320, 2024: 0.122), against Nubank, Inter and C6 at 0.031 (C6 is 440 of the 723 disputes, after its conditional fee from 2023-04);
    - big-five median listed Package I: R$11.55 (2016) and R$15.95 (2024), nominal.
  - The Fees agent's record is `DATA_RAW/_requests/fees_2026-10-23/STATUS.md`.
- **Fees agent, state at 14:00:**
  - Every task is closed except F5 (on hold), F6 (needs Pedro's browser downloads) and a possible LAI for BCB complaints by type.
  - Outputs are in `Open-Finance/BCB/Tarifas/processed/`; the status file is `DATA_RAW/_requests/fees_2026-10-23/STATUS.md`.
  - The consumidor.gov.br complaints are now mapped to 85 conglomerates (94.7% of complaints). Panel-ready side tables exist by conglomerate × quarter and by conglomerate × municipality × year.
  - No keys, no paper text, no pipeline change.
- **Scraping agent:** no tasks left. It waits on the open LAIs (CCS 18810.025400 and its Ouvidoria, 025399, 025440, BB 18882.002114) and Pedro's tariff follow-up (NUP not recorded).
- **K8 (Pedro, 10-27): keep the national g fallback as headline.** After R2, the firm-type fallback goes in a footnote as a robustness check, quoted through keys.
- **PicPay, main-thread proposal (not yet decided):**
  - R2 runs on the current panel.
  - Three cheap local checks: PicPay's share of type-5 balances; whether C0088022's 2022Q3 appearance is a D entry event; the 2022Q3 demand residual of C0080903.
  - Then either keep the data with an appendix sentence, or re-map PicPay as one firm (C0088022 in every quarter) before TR5 is staged, with a re-run of the main estimates.
  - Defaults until then: keep the 102% rule on C0080903's wallet balance, and leave the 2022Q3 switch as data.
- **Housekeeping done (main thread, 14:30):**
  - Temp: `fee_probe`, `scrape_FEE`, `scrape_T5`, `scrape_E2`, `edel_check` and `findex_check` deleted;
  - the empty `Code/Egan_et_al_2025_Rep/other/` and `Rout/followups_cache` deleted;
  - `DATA_RAW/issuers/posted_rates/evidence/iti_termo_20201125.html` (a "page not found" capture) deleted. Its source-log row still says "29 distinct files saved", so it is now 28.
  - The BCB zips had already left Downloads.

## 2026-10-21, later. The code now follows the rename

- **Figures and tables serve both papers.** `V_Full.tex` and `V_Friction.tex` input the same exhibits (8 figures, 24 `\input` files, `paper_numbers_macros.tex`), all flat in `Drafts/Deposit Competition` with no `\graphicspath`. Every generator writes each exhibit once into that folder, never per paper.
- **Which paper the scripts read is ONE setting: `Code/Egan_et_al_2025_Rep/config/paper.toml`** (`paper = "V_Full"` now), read through `utils/paper.py`. It is Pedro's choice; do not change it without asking him.
  - The setting covers `make_paper_equations.py`, `make_paper_numbers.py` (the routine enumerate order, which V_Friction shares) and `diagnostics/report_entry_paths_estban.py`.
  - For one run: `--paper V_Friction` on the equations and report scripts, or `DEPOSIT_PAPER=V_Friction` for all three.
  - `paper_equations_macros.tex` is regenerated, so it names V_Full.
- **Still naming V_Main:**
  - comments and docstrings ("V_Main eq 16" means the same equation in V_Full);
  - the hash-pinned files `bbl_solve.py`, `make_bbl_cost_tables.py`, `utils/sleep_links.py` and `utils/switch_cost_sleep_links.py`, left alone so the bundle checks keep passing;
  - the deck line `\PEQpaperpdf`, which is Pedro's (step 12 of `deck_edits_2026-10-30.tex`).
- **The lit-review subfiles** (`sec_lit_review_full.tex`, `sec_lit_review_friction.tex`) are Pedro's to finish.

## 2026-10-21. The paper file was renamed, and a new working paper exists

- **`V_Main.tex` is now `V_Full.tex`**, in `Drafts/Deposit Competition`. It is the full paper, including the cost estimation, the Egan et al. (2025) replication for Brazil, and its extensions.
  - Any path, script, macro or subfile that points to `V_Main.tex`, `V_Main.aux` or `V_Main.pdf` must now point to `V_Full`.
  - Do not rename or delete `V_Full.tex`.
- **`V_Friction.tex` is new**, in the same folder. It is where Pedro works from now on. Edit plans for paper text target `V_Friction.tex` unless Pedro says otherwise.
- **Known references to the old name**, as of 2026-10-21. These need updating, by Pedro or by the agent that owns each file:

```
refs[4]{file,line,reference,owner}:
  Drafts/Deposit Competition/paper_equations_macros.tex,2 and 28,"generated from V_Main.tex/.aux, \PEQpaperpdf{V_Main.pdf}",coding agent (make_paper_equations.py)
  Drafts/Deposit Competition/sec_lit_review.tex,1,"\documentclass[V_Main]{subfiles}",Pedro (decide V_Full or V_Friction)
  Drafts/Presentations/Deposit Competition/2026-10-30/Presentation-2026-10-30.tex,72,"\PEQpaperpdf{../../../Deposit Competition/V_Main.pdf}",Pedro (deck edit)
  Code/... make_paper_numbers.py, make_paper_equations.py, and any script that reads the paper's .aux or .tex,—,unknown until checked,coding agent
```

- **Unchanged rules:**
  - Claude never edits the paper or the decks. Pedro applies every change from edit plans.
  - Numbers come from `\pn` keys only.
