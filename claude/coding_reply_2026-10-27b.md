# Coding reply: E1 rebuild done, R3 running

**To:** main thread, for Pedro. **From:** coding agent. **Written:** 2026-10-27, 17:40.

## Summary

- **E1 is done.** The production panel is now **`d01b8bae`** (it was `0c47f30a`): `panel_pipeline.py --from 3 --skip 38 --fees`, exit 0. Every audit passes.
- **R3 is running: job 31020411** (Sleep + Logit + RC-BLP, `--no-bbl --no-cf`, on `d01b8bae`, with calendar lags).
- **Paper numbers refreshed with `--cluster-pending`:** 372 entries; crosscheck 155 match, 0 mismatch. 39 panel-family ids changed (below). Cluster-family values stay pending until R3.
- **FK is done:** `collect_fees`, its own "local" family, never pending. The 10 `fees.*` keys equal the handoff's values: POF 0.277 / R$38.19 / R$10.59; disputes big5 0.249 (n 194,895), digital 0.031 (n 23,587), big5 2024 0.122; Package I medians R$11.55 (2016) and R$15.95 (2024). The definitions are copied from the sidecar into `paper_numbers.md`. No other value moved because of FK; the moves below come from the E1 panel.

## What E1 changed (all Pedro's decisions, 10-27)

| Change | Where | Effect |
|---|---|---|
| ESTBAN cells that did not file (`AGEN_PROCESSADAS = 0`) | `utils/estban.py`, step 14 | **Gap** (filed earlier in the quarter and files again later) takes the nearest filed month of the quarter: 18 cells, including **C6 2024Q1** and **BTG 2019Q4**. **Closure** (never files again; 2025–26 used only to decide) is an exit: 79 cells, including the 18 Santander cells of 2024Q4. **No donor** stays missing: 43. Record: `PANEL_INTERMED/estban_unfiled_cells.csv`. |
| B/D rule | step 18 | No retail-deposit test; snapshot **2024-12** (was the latest month, 2026-03). PagBank, C6 Consignado and Letsbank → D; Banco Arbi → B (it sits just under the size cutoff in 2024-12). |
| Type-5 allocation | step 14 | A firm with no ESTBAN checking deposits spreads its IF-Data type 5 by total deposits; one with no ESTBAN deposits at all spreads it evenly over its rows. PagBank's R$1.4–7.0 bn a year enters. |
| Tier rule | step 14 | Decided per firm on 2016–2024. Rodobens, Randon and Andbank come back as national rows. |
| **PicPay as one firm, C0088022** (route 2) | `utils/firm_remaps.py`; steps 14, 16, 17; `config/type5_balance_rates.csv` | C0080903's prepaid for 2019Q4–2024Q3 moves to C0088022, and so do PicPay's posted-rate rules (17 rules, 8 conglomerates). C0088022 takes C0080903's consolidated characteristics through 2024Q3 (from 2014Q4), so its cells stay in the demand sample: characteristics in 35 of 36 quarters (was 0). |
| Calendar lags | `utils/calendar_lags.py`; `sleep_est_e2.py` and the demand preps | A row whose previous quarter is missing gets no lag (374 long rows), instead of a lag from 2+ quarters back. Ships with R3. |

**Audits:**
- MCA strict 9/9;
- firm class all pass (0 type/market disagreements);
- `demand_chars --check` **29/29**, with the cell-for-cell raw comparison now including the filled cells;
- fee audit pass;
- state centering tier 0 15/15.

**Panel before → after (2016–2024 sums of quarters):**
- 500,186 → 500,439 rows; 536 → 537 firms;
- a4 R$62,877 → 62,964 bn; a5 R$736.8 → 761.8 bn;
- rows right after a hole 94 → 87.

**Entry events:**
- B kept 201 → 201.
- **D kept 63 → 67:** PagBank, plus Rodobens, Randon and Andbank. The last three are a local → national switch of source at 2016Q1; Pedro chose to keep them as events.

## Paper-quoted keys that moved (names unchanged)

| Key | Old | New |
|---|---|---|
| `desc.t1.spread5.D.mean` | 0.596 | 0.586 |
| `slide.types.prepaid.share` | 0.520 | 0.536 |
| `slide.types.prepaid.spread.D` | 1.878 | 1.851 |
| `type5.rules.share_posted` | 0.258 | 0.256 |
| `slide.motiv.dpremium.cdi_balance_share` | 0.073 | 0.067 |

`type5.rules.n_zero` (7) and `share_zero` are unchanged at 3 decimals.

## Other panel-family moves worth reading

- **Sign changes (3 leaves), all in `slide.motiv.anova` comove.D:** the D-firm co-movement at h = 4 goes from 0.004 to −0.024 (corr) and from 0.031 to −0.201 (slope). It is insignificant both times. The D set changed (PagBank, C6 Consignado and Letsbank in; Arbi out), and PicPay is one firm.
- **`desc.t2.log_dep`** (pre/post log deposits): the difference goes from −0.252 (p 0.381) to −0.015 (p 0.959).
- **`desc.t2.log_dep_a4`**: the % change goes from 259.6 to 364.4.
- **`desc.fees`**: the digital change 2019–2024 goes from −0.535 to −0.945 (the digital set grew).
- **`desc.t1.spread4.B`**: the mean goes from −0.005 to −0.040, n from 2,145 to 2,089.
- **`desc.t1.spread4.D`**: the mean goes from −1.362 to −1.312, n from 2,377 to 2,543.
- **`desc.a4.n_d`**: +4 to +6 D firms a year.
- **`slide.motiv.reprice_d` / `reprice`, `pix.exposure.*`**: small moves; some significance stars change in the repricing and anova sidecars.

The full diff is in my scratchpad (`e1_pn_diff.txt`), available on request.

## Exhibits

- **Local exhibits refreshed on `d01b8bae`,** except the **entry-path charts**: the `entrygap` key, and the `paper`, `slidesize`, `slidesize2` and `ref` sets that include it.
  - These rebuild the entry paths and refuse to draw when the result differs from `d6_entry_paths.csv`. That file is the entry step's output on the old panel (difference 0.002).
  - R3's entry step writes the new one, so they redraw after the R3 ingest, together with the `slide.motiv.entrygap.*` keys.
- **P1 re-run on the new panel:** 2,552 rows. With PicPay as one D firm and PagBank D, the D share of PF keys is 0.441 (2020Q4) → 0.594 (2024Q4). **D passes B in 2022Q3** (it was 2023Q4). C6 is back in 2024Q1 (the in-panel share is 0.961).

## Paper text this needs (main thread; V_Friction)

1. **Data appendix, ESTBAN:** a branch that did not file is not a zero. Gaps take the bank's nearest filed month of the quarter; closures are exits.
2. **Data appendix, firm type:** the B/D rule no longer tests retail deposits, and it is judged at 2024-12.
3. **Type-5 text (D3):** PicPay is one firm (C0088022) in every quarter, including its wallet before 2024Q4 and its rules. The quoted keys keep their names.
4. **Data appendix, Step 3 (type-5 allocation):** total-deposit and equal weights apply when a firm has no checking footprint.
5. **Methods:** lags are by calendar quarter.

## Next

- **R3:** monitor; when it returns, ingest, run `exhibits_post_cluster.py --run` and the entry-path charts, then plain M1 and the changed-key report.
- **After R3:** K8 (firm-type g fallback, footnote keys), then TR5 staging on `d01b8bae`.
- **Housekeeping later:** the scratchpad holds the pre-E1 copies (`e1_before`, about 0.5 GB) until R3 is ingested.
