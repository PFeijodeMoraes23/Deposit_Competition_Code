# Handoff: bibliography tasks from the October 20 plans

**To:** bibliography agent. **From:** main thread, for Pedro. **Date:** 2026-10-20.

**Rule:** `V_Main.tex` and the decks are edited by the author only. This brief asks for `References.bib` entries and checks only.

## Basis

```
files[4]{file,role,saved}:
  V_Main.tex,paper,10-19 10:42
  2026-10-20Devo/V_Main_followups_plan_2026-10-20.md,plan citing B1 (block V10) and B2 (block V4),10-20
  2026-10-20Devo/devo_followups_plan_2026-10-20.md,plan listing the same references,10-20
  Drafts/Deposit Competition/deposit_types_report_2026-10-15.md,Appendix C lists the regulatory keys added 10-12 and 10-13,10-15
```

- **`References.bib`** sits in `Drafts/`, which is outside the main thread's connected folders. That is why B2 is a check and not a claim.
- **Numbering.** This brief replaces the bibliography lists of both plans. The list in `devo_followups_plan_2026-10-20.md` used B2 to B4 for other items. Use the numbers below.

## Tasks

```
tasks[5]{id,agent,entry,status,blocks}:
  B1,bib,"Agarwal and Somaini, Demand Analysis under Latent Choice Constraints",new,V10
  B2,bib,"confirm nine regulatory keys used by block V4",check,V4
  B3,bib,"Adao, Kolesar and Morales (2019), Shift-Share Designs: Theory and Inference, QJE",new,"later V8, code task C4"
  B4,bib,"Borusyak, Hull and Jaravel (2022), Quasi-Experimental Shift-Share Research Designs, RES",new,"later V8, code task C4"
  B5,bib,"Resolucao BCB n. 19/2020 (Pix fees for individuals)",new,"deck Pix primer, optional V5"
```

### B1. Agarwal and Somaini

- **Authors:** Nikhil Agarwal and Paulo J. Somaini.
- **Title:** "Demand Analysis under Latent Choice Constraints".
- **Versions found:**
  - NBER Working Paper 29993 (April 2022, revised May 2025)
  - a *Review of Economic Studies* manuscript dated October 2025.
- **Published version.** The NBER page lists *Review of Economic Studies*, volume 93, issue 4, pages 2215 to 2249.
- **The year is unresolved.** Volume 93 of the *Review of Economic Studies* should be the 2026 volume, while Pedro recalls 2025.
  - Take the year, volume, issue, pages and DOI from the journal's own page.
  - If the issue is not out yet, use the online-first entry and say so.
- **BibTeX:** the Google Scholar BibTeX, verbatim, corrected only against the journal page.
- **Key:** `agarwal20XXdemand`, with the verified year.
- **Where it is cited:** block V10 of the V_Main plan, Section 6.3 (Taxonomy of the friction), V_Main L1040. The block currently holds the placeholder `\textcite{KEY_B1}`. Report the final key so the main thread can update the block.
- **Sources:**
  - https://www.nber.org/papers/w29993
  - https://www.restud.com/wp-content/uploads/2025/10/manuscript.pdf
  - https://economics.mit.edu/research/publications/demand-analysis-under-latent-choice-constraints

### B2. Confirm the keys of block V4

Block V4 adds a paragraph on the legal regime of each deposit type to Section 2.2 (Deposit Types), V_Main L275. It cites the nine keys below, which `deposit_types_report_2026-10-15.md` (Appendix C) reports as added on October 12 and 13.

For each key, confirm two things:

- the key exists in `References.bib`
- its entry matches the act named in the "Supports" column.

```
keys[9]{key,supports}:
  brasil_lei8981_1995,savings income exempt from income tax (art. 68 III)
  brasil_lei11033_2004,income-tax table on fixed income by holding period (art. 1)
  brasil_decreto6306_2007,IOF on redemptions within 30 days (art. 32 and annex)
  cmn_resolucao_5005_2022,time deposits cannot renew automatically (art. 4 I c)
  bcb_compulsorios_historico,reserve-requirement rate history
  bcb_resolucao_145_2021,reserve requirements on time deposits (remunerated at Selic)
  bcb_resolucao_189_2022,reserve requirements on demand deposits
  bcb_ee72_2020,directed credit of demand deposits (BCB Estudo Especial 72/2020)
  cmn_resolucao_4676_2018,housing credit from savings (SBPE)
```

Report any key that is missing, misspelled, or points to a different act. Do not rename existing keys.

### B3. Adão, Kolesár and Morales (2019)

- **Reference:** Rodrigo Adão, Michal Kolesár and Eduardo Morales, "Shift-Share Designs: Theory and Inference", *Quarterly Journal of Economics*, 2019.
- **BibTeX:** the Google Scholar BibTeX, verbatim, checked against the journal page for volume, issue, pages and DOI.
- **Key:** `adao2019shift`.
- **Use:** inference for the mobile-access Pix exposure (code task C4, and a later extension of block V8). Not cited yet.

### B4. Borusyak, Hull and Jaravel (2022)

- **Reference:** Kirill Borusyak, Peter Hull and Xavier Jaravel, "Quasi-Experimental Shift-Share Research Designs", *Review of Economic Studies*, 2022.
- **BibTeX:** the Google Scholar BibTeX, verbatim, checked against the journal page.
- **Key:** `borusyak2022quasi`.
- **Use:** same as B3. Not cited yet.
- **Duplicates:** check that neither B3 nor B4 is already in the file under another key. `goldsmithpinkham2020bartik` exists and is a different paper.

### B5. Resolução BCB nº 19/2020

- **The act:** the Banco Central do Brasil resolution of 2020 that sets the Pix fee rules, including free transfers for individuals.
- **Form:** an entry in the same style as `bcb_resolucao_1_2020` and `bcb_resolucao_80_2021`. Take the date, number and URL from the official text at normativos.bcb.gov.br.
- **Key:** `bcb_resolucao_19_2020`.
- **Use:** the deck's Pix primer, where it is cited in a comment for now, and optionally block V5 (Section 2.1).
- **Current consolidated text.** If the act was amended after 2020, note which version the entry points to.
- **Source found, not the act itself:** BCB Vote 269/2020, https://normativos.bcb.gov.br/Votos/BCB/2020269/Voto_do_BC_269_2020.pdf

## Acceptance

- **File format** is preserved: CRLF line endings and UTF-8.
- **Braces** balance and **no key collides** with an existing one.
- **A test document** with V_Main's bibliography options, citing B1, B3, B4, B5 and the nine B2 keys, gives zero biber and LaTeX warnings.
- **V_Main's own output** is byte-identical before and after, since none of the new keys is cited there yet.
- **Report** the final keys, the entry count before and after, and anything not found.

## Honesty flags

- The B1 year is unverified. Do not write 2025 or 2026 without the journal page.
- B5 must come from the official BCB text, not from press or bank pages.

## Open questions for Pedro

1. If B1 is out only as online-first, should the entry use the NBER working paper instead?
