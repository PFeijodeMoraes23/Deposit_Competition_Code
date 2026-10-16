# Handoff: typed numbers and claims now stale in the motivation memo and the two decks (2026-10-16)

For the writing agent (or Pedro). Every item is typed text. The generated exhibits and the `\pn` values are already current. No file below has been edited.

**Basis.**
- `Drafts/Deposit Competition/motivation_figures_review.md`, saved 10-16 13:58. Line numbers below refer to this save.
- Decks: `Drafts/Presentations/Deposit Competition/2026-10-06/Presentation-2026-10-06.tex` and `.../2026-10-19-IOTea/Presentation-2026-10-19.tex`. Both were saved 10-05 19:01 and are identical in the lines below.
- Values come from:
  - the exhibits regenerated today at 14:17–14:27: `tab_slide_deposit_types.tex`, `Compressed_*`, `tab_motiv_*`;
  - the sidecar `Rout/slide_motivation.json`.
- **`paper_numbers_macros.tex` is still the 00:25 file.** The refresh is blocked: `config/routines.toml` (10-15 20:04) now names the spec-12 column `phi_t_IV_RivalSpread_x_Tech`, while the sleep outputs on disk (09-14) still carry `HausmanFull`. The refresh has to wait for the sleep rerun.
  - Until then, these keys hold their 00:25 values: `slide.types.*.share`, `slide.motiv.prepaid.*`, and the 2020 cells of `desc.a4.*`.
  - In V_Main none of them changes what is displayed: the shares print to one decimal, and V_Main uses no 2020 key.

**Background to the changes.**
- No institution-level prepaid (type 5) rate is observed inside 2016–2024, so every prepaid balance carries the CDI rate and the prepaid spread is the Selic minus CDI gap. The details are in `utils/prepaid.py` and in the V_Main plan `claude/prepaid_fallback_vmain_plan_2026-10-16.md`.
- Prepaid balances now start in **2020Q2** in every exhibit, as in the data. Before today, the motivation charts started them in 2020Q4.

---

## A. Motivation memo (`motivation_figures_review.md`)

| ID | Line | Now says | Should say | Source |
|---|---|---|---|---|
| M1 | 52–53 | "Time deposits and prepaid accounts are priced by banks." | Time deposits are priced by banks. Prepaid accounts are too in principle, but no institution-level prepaid rate is observed in the sample, so every prepaid balance is assigned the CDI rate. | `utils/prepaid.py` |
| M2 | 59 | "This happens in 11 percent of B-firm quarters and 60 percent of D-firm quarters (9 percent of D-firm time-deposit balances)." | B: 7.9 percent of all B conglomerate-quarters (186 of 2,349), or 7.1 percent of those holding time deposits (152 of 2,145, the 2,145 → 1,993 step of the waterfall). D: 60.2 percent of all D firm-quarters (`dpremium.d_rowshare_with_cdi_rate` = 0.602), or 33.8 percent of those holding time deposits. The 9 percent of balances is still right (0.085). **The current sentence mixes bases.** Use one base for both firm types; the time-deposit base is the natural one. | B shares and D 33.8%: no key, computed by the audit; D 60%: sidecar |
| M3 | 166 (Fig. 2 caption) | "The light band around each of these three lines … is its 95 percent interval." | The prepaid band has zero width (`ci_halfwidth_mean.prepaid` ≈ 5e-21), so only the two time-deposit lines show a band. Say the prepaid line is the CDI fallback and carries no band. | sidecar |
| M4 | 168 (Fig. 3 caption) | "Prepaid accounts start with Pix in 2020Q4." | "Prepaid accounts start in 2020Q2, when their balance is first reported." The chart now starts there (`slide.motiv.prepaid.first` = 2020Q2 once the macros refresh). | sidecar `pricevol.prepaid.first` |
| M5 | 170 (Fig. 4 caption) | "the light bands are the same 95 percent intervals" | Same as M3: no band on the prepaid line. | sidecar |
| M6 | 199–203 (Table 2 notes) | "In these quarters nearly all are priced at the CDI" | "In these quarters all are priced at the CDI". Every type-5 rate in 2020–2024 equals the CDI. The table rows now read "Prepaid accounts, B/D firms, bp (CDI rate)". | panel check, `tab_motiv_tightening.tex` |
| M7 | 580–581, 612, 636, and later | Santander mentions in the entry-by-gap section | **Decision for you.** The numbers are correct: 201 kept entries, Santander 122, split 107/94 on the 201. On 10-15 the Santander section was removed and these passages were kept. Separately, the 107/94 split (all 201 entries) sits near the 106/92 split (the 198 with a usable path) that the table and figures print. If both stay, name the base each time. | `d6_entry_*` |
| M8 | appendix key table | generated from `paper_numbers.json` | Re-render it after the macro refresh. Until then the `slide.motiv.prepaid.*` rows are the 00:25 values (first = 2020Q4). | — |

Already fixed in the 13:58 save: "about 2.5 percent a year" at L357 and L658 now matches `relprice.share.xs_pp` = −2.509.

**Also new today, for the memo text:**
- The D-firm repricing charts (`fig_slide_motiv_reprice_binscatter_d`, `_points_d`) now say "firm-quarters of N firms" instead of "bank-quarters of N banks".
- The deposit-types table now marks the prepaid rate setter "Bank†" and adds a line under the table: "No bank-level prepaid rate is observed: the CDI fallback, so the spread is Selic − CDI."

---

## B. Decks: speaker-note comments on frame S6, `frame:deposit-types` (L305–318 in both files)

All of these are `%` comments. The table and the `\pn` calls on the slide are current.

| ID | Line | Typed now | Value now | Key |
|---|---|---|---|---|
| D1 | 308 | "1.3 pp … for B firms and -1.1 … for D firms" | 1.4 and −1.3 (1.392, −1.307) | `slide.types.time.spread.B/D` |
| D2 | 309 | "Prepaid accounts, bank: 0.000 … and 0.001 … for D firms" | 0.000 and 0.000. Add that this is the CDI fallback (the table now carries a † note). | `slide.types.prepaid.spread.B/D` |
| D3 | 309 | "In Table A.4 it rounds to 0.00 in every year from 2020, at most 0.002 … (D firms, 2024)" | 0.000 in every year, D 2024 included. Since 16:20 Table A.4 no longer prints the prepaid columns; the key still exists (it comes from the CSV), so cite the key rather than "Table A.4". | `desc.a4.spread5.D.2024` |
| D4 | 311 | "Backup … -0.07 … for B firms and -0.92 … for D firms" | −0.041 and −0.847 (Table 1) | `desc.t1.spread4.B/D.mean` |
| D5 | 318 | Say-line: "1.3 pp … at branch banks" | 1.4 | `slide.types.time.spread.B` |
| D6 | 309 | "5 Prepaid accounts, bank" | Suggest "5 Prepaid accounts, bank in principle; in the data, the CDI fallback". The table row now reads "Bank†". | — |

Checked and still right:
- demand 9.0 and savings 3.2 (9.021, 3.212);
- Selic 14.3 (2016) and 2.8 (2020);
- shares 8.4, 32.6, 58.5 and 0.5 (now 8.405, 32.560, 58.515, 0.520; the same at one decimal).

---

## C. Items for other agents

1. **Main thread.** Refresh `paper_numbers_macros.tex` after the sleep rerun that produces the `RivalSpread` columns. Then re-render the memo's key table (M8).
2. **Generator, optional.** M2 has no key for the B-firm CDI-fallback share, nor for either firm type's share among firm-quarters holding time deposits. If the memo keeps these numbers, the dpremium sidecar in `make_slide_figures.py` could record them, and `collect_slide_motivation` would publish them.
