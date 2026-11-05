# Pix handoff, 2026-11-04 (20:30): reopened for one rerun on the I14 panel

**Rules** as in `pix_handoff_2026-11-03c.md` (closed 17:50): read the panel with pyarrow by column subsets, record the md5 in every output, give Remove-Item lines for anything outside `Rout`, reply in a dated file.

**Basis.** The coding thread's I12 (`52ea6495`, 18:34) and I14 (`0b9385b5410b5cb6ae4c884c32e4ef46`, 20:02) replace `45dbfc58`, the panel your 17:25 reruns used. I14 re-carried ten benchmarked quarters (Itaú 2016Q1 to Q4, 2018Q1 to Q2, 2020Q1 to Q2; Banrisul 2020Q1 to Q2) whose rates lay outside 0.5 to 1.5 times the CDI; 153 imputed tails remain and are decided on 11-05. Rows, firms, markets and deposit sums are unchanged.

```
tasks[2]{id,what,status}:
  P6,"rerun P3, P4 and P5 on 0b9385b5 (set K0_PANEL_MD5=0b9385b5; demand_chars side file records it); same two lines as at 17:14; outputs overwrite in Rout with the md5 in every row; p3_note.md and p4_note.md get a section on the rerun with the before and after table (45dbfc58 against 0b9385b5)",new
  P7,"write the P3 first-stage numbers the paper's keys read from p3_note.md's table (median S, F, placebo F, F net of structure, by panel) to Rout/p3_numbers.json, so make_paper_numbers.py (collect_pix_stakes) can read them from a file; keep the markdown table as well",new
```

Expected: six minutes for P6. One more rerun will follow the imputed-tail decision of 11-05, on the md5 the coding thread posts then. Reply in `PIX_REPLY_2026-11-04d.md`.
