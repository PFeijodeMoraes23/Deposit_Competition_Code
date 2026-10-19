# Reply to request 7: reference-style `_ref` files for the October 20 deck

Done. All six files are in `Drafts/Deposit Competition`. Each is drawn at its final size, so place it at natural size, with no `width=`.

| ID | File | Written | Size (target) | Smallest text | Deck frame |
|---|---|---|---|---|---|
| R1 | `fig_slide_motiv_spreads_by_type_ref.pdf` | 18:59 | 6.29 × 4.62 cm (6.3 × 4.7) | 8.0 pt | `frame:motiv-tightening` |
| R2 | `fig_slide_motiv_shares_by_type_ref.pdf` | 18:59 | 6.28 × 4.61 cm (6.3 × 4.7) | 8.0 pt | `frame:motiv-tightening` |
| R3 | `fig_market_structure_hhi_ref.pdf` | 18:59 | 10.75 × 3.77 cm (10.8 × 3.8) | 8.0 pt | `frame:two-kinds` |
| R4 | `fig_slide_motiv_dfirm_premium_ref.pdf` | 18:59 | 6.48 × 4.94 cm (6.5 × 5.0) | 8.0 pt | `frame:motiv-entry`, left column |
| R5 | `fig_slide_relprice_decomp_time_ref.pdf` | 18:59 | 10.81 × 4.54 cm (10.8 × 4.6) | 8.5 pt | `frame:b-periods` |
| R6 | `fig_slide_motiv_entry_by_gap_bankboot_ref.pdf` | 19:00 | 10.74 × 3.84 cm (10.8 × 3.9) | 8.5 pt | `frame:entry-channel` |

**Acceptance.**
- Every file is within 0.1 cm of its target.
- No text is under 8 pt.
- No label crosses data.
- The originals and all earlier slide files are unchanged.
- `Rout/slide_motivation.json` is unchanged (md5 `52f1ffcd…`), so the end values come from the same series as the originals. For example, D's share of time deposits ends at 15.1%, against `slide.motiv.dpremium.d_share_time.end` = 15.126.

**The reference look.**
- Title at the top left.
- Thin lines (1.2 pt at print size).
- End-of-line values in each line's colour, spaced so they don't touch.
- The grey "Selic hikes" band, with a little headroom so its label clears the top line.
- Legend below the plot.
- No markers.
- Same data, samples, weights, intervals and colours as the originals.
- R1 and R2 keep the prepaid line and the 95% bands, as in the originals.

**Changes beyond size, per file.**
- **R1, R2:**
  - The legend has two columns; "Prepaid (from 2020Q2)" breaks onto two lines.
  - R2's title breaks onto two lines ("Share of balances / by deposit type"), because one line is wider than 6.3 cm.
- **R3:**
  - No Pix line or label, and no markers.
  - The 1,500 and 2,500 rules and their labels stay.
  - The legend sits inside the plot on the right, between the two lines, clear of both and of the 1,500 label.
  - The "Year" axis label is dropped to keep the plot height.
- **R4:**
  - Two stacked panels with a shared x axis.
  - One-line titles: "D premium over B (95% band)" and "D share of time deposits". Each starts over its y label, since one line is wider than the plot.
  - The "95% interval" legend is dropped, since the title names the band.
- **R5:** to fit two legend rows in 4.6 cm, the legend names are shortened to "Common to all banks", "Persistent, B", "Rest, B", "Persistent, D" and "Rest, D". The bars and values are unchanged.
- **R6:**
  - The legend is at the right, with the counts: "Pays ≥ 1 pp more: 106 entries, 8 banks" and "Pays < 1 pp more: 92 entries, 14 banks".
  - No markers.
  - The y label breaks onto two lines.

**Regenerate.** Run `python make_slide_figures.py --only ref --png`. Only these six files are written; the sidecar is not touched.
