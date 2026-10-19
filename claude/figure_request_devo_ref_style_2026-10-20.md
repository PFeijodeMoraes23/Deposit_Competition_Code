# Request 7 to the figure generator: reference-style slide files for the October 20 deck

**Date:** 2026-10-20. **From:** main thread, for Pedro. **Deck:** `Presentations/Deposit Competition/2026-10-20Devo/Presentation-2026-10-20.tex` (self-guided talk, the slides are not shown to the audience).

**Reference style.** The paper-style files of 10-16 14:20, for example `fig_slide_motiv_spreads_by_type.pdf` and `fig_slide_motiv_shares_by_type.pdf`:

- title at the top left
- thin lines with end labels in the line's color
- the gray "Selic hikes" band
- legend below the plot
- no markers.

Pedro prefers this look to the `_slide` style (thick lines, markers). Keep the same data and colors. Draw each file at its final size with text of at least 8 pt, so the deck places it at natural size with no `width=`. Originals and earlier files stay unchanged. Use the suffix `_ref`.

**Interim.** Until these land, the deck scales the paper-style files, so their text prints at about 4 to 5 pt.

```
files[6]{id,new_file,from,size_cm,changes}:
  R1,fig_slide_motiv_spreads_by_type_ref.pdf,fig_slide_motiv_spreads_by_type.pdf,6.3x4.7,"none beyond size. Two per row across the page width"
  R2,fig_slide_motiv_shares_by_type_ref.pdf,fig_slide_motiv_shares_by_type.pdf,6.3x4.7,"same as R1, so the pair matches"
  R3,fig_market_structure_hhi_ref.pdf,fig_market_structure_by_year_a.pdf,10.8x3.8,"no Pix line or label. Keep the 1,500 and 2,500 rules"
  R4,fig_slide_motiv_dfirm_premium_ref.pdf,fig_slide_motiv_dfirm_premium.pdf,6.5x5.0,"two stacked panels, shared x axis, one-line titles"
  R5,fig_slide_relprice_decomp_time_ref.pdf,fig_slide_relprice_decomp_time.pdf,10.8x4.6,"none beyond size"
  R6,fig_slide_motiv_entry_by_gap_bankboot_ref.pdf,fig_slide_motiv_entry_by_gap_bankboot.pdf,10.8x3.9,"legend at the right with the entry and bank counts, as in the _slide2 file"
```

**Where each file goes in the deck.**

- R1 and R2: `frame:motiv-tightening`.
- R3: `frame:two-kinds`.
- R4: `frame:motiv-entry`, left column.
- R5: `frame:b-periods`.
- R6: `frame:entry-channel`.

**Acceptance.**

- Each file is at its stated size, within 0.1 cm.
- No text is below 8 pt.
- No label crosses data.
- End values match the macros of `paper_numbers_macros.tex`, as in the 12:30 reply of October 19.
- Report the file names and times to the main thread.
