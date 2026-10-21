# Shared memory for every Claude instance on Pedro's machine

Read this before any task that touches the paper. Newest entries first.

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
