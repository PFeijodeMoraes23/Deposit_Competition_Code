---
name: bib
description: Edit or check the shared bibliography Drafts/References.bib - add or correct entries, repair broken or renamed citation keys, apply the Econometrica house style with the user's overrides, cite Brazilian laws and BCB, CMN and INSS acts correctly, and verify the result with biber. Use whenever a task touches References.bib, a citation key, the formatting of a reference, an unresolved-citation warning, or a legal or regulatory act that needs citing.
---

# The shared bibliography

`Open-Finance/Drafts/References.bib` (from this repo: `../../Drafts/References.bib`) holds about
800 entries and is shared by every draft: the deposit paper, the literature reviews, the
presentations, the labor-market papers. A change made for one paper is a change for all of them,
which is where most of the rules below come from.

## File handling

- UTF-8 with CRLF line endings; keep both. Python's text mode converts CRLF to LF on read and
  write: open with `newline=''` and write `\r\n` explicitly.
- Literal accented characters are fine; every draft loads `inputenc utf8`.
- Record the entry count before and after, and check that braces stay balanced after any
  programmatic edit.

## Keys

- Never rename or remove a citation key. Other `.tex` files cite it, and a past cleanup that
  renamed keys broke live citations in a literature review, three presentations and the deposit
  paper without a single error.
- When two entries must become one, or a key has to change, keep the old key alive as an alias on
  the surviving entry: `ids = {oldkey}`. Biber resolves both. The same field merges a working
  paper into its published version. An alias must not equal another entry's key.

## What biblatex and biber need

Every active draft uses biblatex with biber.

- `@legislation`, `@legal` and the `jurisdiction` field have no driver and render broken. Use
  `@misc` with `howpublished`. The `publisher` field is not printed for `@misc`.
- `month` is an integer. Biber warns on `nov` and on Portuguese abbreviations.
- A corporate author takes double braces: `{{Brasil}}`. With single braces biber parses "Instituto
  Nacional do Seguro Social" as a person whose surname is Social.

## House style: Econometrica, with the user's overrides

The authority is the Econometric Society's current template (`econsoc.bst` and its sample PDF at
`github.com/vtex-soft/texsupport.econometricsociety-ecta`), not a remembered older style. Its
reference format: `Aumann, Robert (1987), "Title." Econometrica, 55 (1), 1–18.` That is full first
names, "and" before the last author even with two authors, the issue in parentheses, no DOI.

The user's overrides, which win over the template:

- Keep biblatex's dash for a repeated author.
- Show access dates, with the day and the month spelled out.
- DOIs stay in the file as data and are hidden in print.
- In text, up to four names before "et al.", with "and" and no serial comma. Keep "(Author, Year)"
  parentheticals and "Egan et al. (2025)"; the template's nested parentheses and initials were declined.

In the entries:

- Titles in Chicago headline case: minor words lowercase unless first, last or after a colon. A
  title-casing pass must skip anything that contains braces, a backslash or `http`.
- Cite the newest version of a paper, after checking that it really was published (see below).
- Status notes (revise and resubmit, conditionally accepted) and the summary of a legal act go in
  `annotation`, which is not printed. `addendum` is printed.
- Do not normalise a journal name by cutting at the colon: that merges *American Economic Journal:
  Macroeconomics* with *Applied Economics*. Only Google Scholar's tail ": Journal of the ...
  Society" is safe to cut.

The preamble block that implements the style goes into the paper by the user's hand. Test it on a
scratch copy and give the lines to paste.

## Brazilian legal and regulatory acts

Authorship follows who signed the act, not who published it:

| Act | `author` |
|---|---|
| Lei, Medida Provisória, Decreto | `{{Brasil}}` |
| Instrução Normativa INSS/PRES | `{{Instituto Nacional do Seguro Social}}` |
| Resolução CMN | `{{Conselho Monetário Nacional}}`; keep "CMN" in the title so it can be found |
| Resolução BCB, Circular, Instrução Normativa BCB, Ato do Presidente | `{{Banco Central do Brasil}}` |
| Resolução Conjunta | both, Banco Central first |

Shape of an entry: `@misc` with `author`, `title` (the official Portuguese form, with a literal
`nº`), `year`, `date`, `howpublished = {Diário Oficial da União}`, `note` (the official ementa,
then the act's status, including a conversion from Medida Provisória to Lei), `url`, `urldate`.

Take the text from the Diário Oficial permalink or a `normativos.bcb.gov.br` PDF. The Banco
Central's `exibenormativo` pages return an empty shell to a fetcher, and a truncated `numero=`
fetches a different act. Past entries carried an invented English title and summaries copied
from a neighbouring act, so read the act itself before writing its entry.

## Adding or correcting a reference

Check each field against the source, the publisher's page or the working-paper series, rather
than against a search-engine snippet. The 2026-08-31 cleanup found wrong first names, a missing
co-author, an affiliation parsed as a person, and a working-paper number that pointed to an
unrelated paper.

Entries left as they are on purpose; do not "fix" them without asking:

- `gandhihoude2019` and `grunewald2020auto` are confirmed unpublished, whatever a search suggests.
- `vogel2022race` and `kogan2021technology`: the right current version is a judgement call.
- `cmn2025resolucao`: its recorded note may be wrong, and whether the act was revoked is unsettled.
- Three keys break the lowercase convention (`MYERS1984187`, `Kroft2020Imperfect`,
  `martinezMiera2010`). Renaming them needs aliases and the user's go.

## Verify

1. Run `biber --onlylog <document>` for each active draft the change can reach and read the
   `.blg`. The target is zero warnings.
2. The `.bcf` reflects only the last LaTeX run, so also walk the document's `\input` chain from
   the source and check every `\cite` key against the keys and aliases in the file. The deposit
   paper's chain has about thirty files.
3. Report the entry count before and after, the keys touched, any alias added, and anything left
   for the user to decide.

Do not edit any `.tex` file to make a citation resolve; an alias in the bibliography does it.
