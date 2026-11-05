# Pix agent reply, 2026-11-04 (answers `pix_handoff_2026-11-04.md`, 20:30)

**Written:** 21:40. P6 and P7 are done.

One finding changes what the paper's `pix.stakes.fs` keys say. The placebo F is one random shuffle, and on `0b9385b5` that one draw is near the top of its own distribution (section 3). I recommend a change, which needs the main thread's OK.

## 1. P6: P3, P4 and P5 on `0b9385b5`

- **Before running.**
  - Panel md5 `0b9385b5410b5cb6ae4c884c32e4ef46`, as posted at the I14 end line.
  - The `demand_chars` side file records the same md5.
  - No panel writer was running.
  - The `45dbfc58` outputs were copied to the scratchpad first.
- **Runs** (pyarrow column subsets, every row stamped `0b9385b5`):
  - `diag_p_shifters.py --only p3`, 21:24–21:25, exit 0.
  - `diag_p4_openings_advertising.py` (P4 and P5), 21:25–21:28, exit 0.
- **Notes.** `p3_note.md` and `p4_note.md` each open with one table, one column per panel (`d01b8bae`, `45dbfc58`, `0b9385b5`). There is one table per note on purpose; see section 2.

**P3:**

```
p3[6]{quantity,45dbfc58,0b9385b5}:
  "S median (25th-75th), R$ a year","49.6 (23.9-85.3)","42.5 (22.4-70.6)"
  "sd of spreads, median, pp","1.00","0.85"
  "log sd on log sd~, F","988","493"
  "same, rho~ shuffled (placebo), F","250","484"
  "same, net of lagged HHI and number of banks, F","709","370"
  "S_measured: market-quarters / corr with S / carried weight","73,390 / 0.996 / 1.6%","72,986 / 0.990 / 3.7%"
```

- Log S correlates 0.85 with `45dbfc58`.
- The type-4 rate chain is the only thing that moved; deposits, rows and markets are unchanged.

**P4 and P5:** nothing moves.
- Advertising: B 0.086 (p 0.004), the rest 0.086, big five 0.063 (p 0.21), difference p 0.50.
- Sums of lags: 0.139 for the rest, −0.043 for the big five.
- The price term is still zero: 0.008, p 0.50; trimmed 0.016, p 0.34.
- P5: IV 0.204 (first-stage F 25.4, AR set [0.053, 0.477]) against OLS 0.086, and group placebo −0.770 (p 0.002).
- **The lagged spread is narrower.** It now runs from −38.2 to +14.8 pp. The big five's 5th percentile rose from −3.9 to −1.9 pp with Itaú's quarters re-carried. The remaining tails are the 153 imputed quarters.

## 2. P7: `Rout/p3_numbers.json`

**What it holds.** A new script, `diagnostics/diag_p3_numbers.py`, reads the p3 block of `p_shifters.json` and files it under the panel's md5 (8 characters, as the CSV's `panel_md5`). It keeps the earlier panels. All three are seeded:

| panel | median S | F | placebo F | net F |
|---|---|---|---|---|
| `d01b8bae` | 98.7 | 246 | 174 | 84 |
| `45dbfc58` | 49.6 | 988 | 250 | 709 |
| `0b9385b5` (`current`) | 42.5 | 493 | 484 | 370 |

- Each entry also carries F in levels and logs, n, the median sd, the cost-fit F, the measured-rate variant where the panel has it, the full md5 and the run time.
- **It checks itself against the CSV.** It refuses to write if `p3_stakes_mca_quarter.csv`'s md5 or median S disagrees with the p3 block.
- **It is a separate script,** so `diag_p_shifters.py` (the Kappa agent's input) is unchanged. After each P3 rerun it is one more line: `python -I diagnostics/diag_p3_numbers.py`.

**For `make_paper_numbers.collect_pix_stakes` (not mine; I did not touch it).**
- **It reads the note correctly today.** I called it alone, which only adds an in-memory entry: median 42.46, f 493, f_placebo 484, f_net 370, no notes.
- **A trap.** It takes the column index from the first table row that contains the md5, then reads every row in the note whose label matches.
  - A second table with the same labels, such as the separate per-rerun tables of 17:25, would overwrite the values without warning, from another table's column.
  - The note now has one table, and each label occurs once.
- **Switching it to `p3_numbers.json["panels"][md5]`,** with md5 the CSV's, removes the parsing. The values are floats there, rounded in the note.

## 3. The placebo F is one draw, and too noisy to quote

On `0b9385b5` the placebo F (484) sat next to the real one (493), which would read as "the cost shifters add nothing". So I ran 20 more shuffles, seeds 1000–1019. They used a scratch copy of the script with one specification added, the placebo net of structure; the script itself is unchanged. Results are in `Rout/p3_placebo_shuffles_0b9385b5.json`.

```
shuffles[2]{statistic,real,placebo_median,placebo_range,shuffles_at_or_above_real}:
  "F, log sd on log sd~","493","158","30-594","1 of 20"
  "F net of lagged HHI and number of banks","370","35","0.8-426","1 of 20"
```

- **The file's shuffle (seed 20261103) is near the top** of that distribution: 484, and 377 net of structure.
- **The cost shifters beat about 95% of shuffles,** a permutation p of about 0.10 with 20 draws. They carry part of the instrument's strength, but much of it comes from the lagged deposit weights that any bank attribute would have.
- **The earlier comparisons are single draws too:** 988 against 250 on `45dbfc58`, 246 against 174 on `d01b8bae`. The 17:25 reading, recorded in the design note, that the instrument is "now carried by the cost shifters", rested on one draw and on Itaú's mismeasured quarters. It should be softened to the line above.
- **Recommendation (needs the main thread's OK).** In `diag_p_shifters.py`, replace the single placebo with 99 or more shuffles, for both specifications (plain and net of structure). Report the placebo median and the share at or above the real F, carry them into `p3_numbers.json`, and have the paper key `f_placebo` quote the median, not one draw.
  - The diagnostic regressions repeat; the CCS and the fit do not. That's a few minutes per run.
  - The change touches the Kappa agent's input file but no column of the stakes CSV.

## 4. Two things for the Kappa agent

- **Panel mismatch.** Step 7 and K1 read the `52ea6495` snapshot, while the stakes file now carries `0b9385b5`; `panel_md5` says so in every row.
  - If K1 must use P3 on `52ea6495`, `diag_p_shifters.py` needs a panel-path option to read the snapshot. That is a small change, and only on request.
  - The two panels differ only in the ten re-carried benchmarked quarters.
- **The `S_measured` row is populated.** Carried rates hold 3.7% of the weight, and corr(log S, log S_measured) is 0.990.

## Housekeeping

- **Scratch.** `remove_pix_scratch_2026-11-04.ps1` in the scratchpad removes the backups and test copies of P3 to P5, with existence checks. Run:

  ```
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\77c31b25-6d18-4e95-8d00-b8a660c0451e\scratchpad\remove_pix_scratch_2026-11-04.ps1"
  ```

  It keeps `compare_rerun.py` for the next rerun. Other folders in the old scratchpad (aw1, r1, ted1, k0 logs) are from the 11-02/11-03 tasks; I left them out.
- **Next rerun.** After the 11-05 imputed-tail decision, on the md5 the coding thread posts: `diag_p_shifters.py --only p3`, `diag_p3_numbers.py`, `diag_p4_openings_advertising.py`, about 7 minutes. Each note gains one column.
