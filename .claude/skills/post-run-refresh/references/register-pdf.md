# Building the advertising register PDF

Master: `ADVERTISING_DATA.md` in the code folder. Copies: `Drafts/Deposit Competition/ADVERTISING_DATA.md`
and `ADVERTISING_DATA.pdf`, next to `V_Main.tex`. After any edit of the master, recopy the md and
rebuild the pdf. Never touch `V_Main.tex`.

Tools: pandoc and MiKTeX's xelatex, both installed.

## Steps

1. Work in a scratch folder so no auxiliary files land in Drafts. Copy `scripts/breaks.lua` from
   this skill into it.
2. Convert (one line):

   `pandoc ADVERTISING_DATA.md -o <scratch>/ADVERTISING_DATA.tex --from markdown-tex_math_dollars-raw_tex --lua-filter <scratch>/breaks.lua --standalone --shift-heading-level-by=-1 --toc --toc-depth=2 -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true --columns=100`

3. Compile in the scratch folder: `xelatex -interaction=nonstopmode -halt-on-error ADVERTISING_DATA.tex`,
   three times when the folder is new. After two passes the contents page still carries the page
   numbers of the pass that had no contents page, about one page early. Stop when
   `ADVERTISING_DATA.toc` no longer changes.
4. Check before copying: extract the text of the old and the new PDF with PyMuPDF, strip
   whitespace, and compare. Only the intended edit and shifted page breaks should differ.
5. Copy the md and only the pdf into the Drafts folder.

## Why each non-default piece is there

- `-tex_math_dollars`: the text is full of "R$...", and one `$` followed by a quote would open a
  math span.
- `--shift-heading-level-by=-1`: the single H1 becomes the title; sections are numbered by hand.
- `breaks.lua`: turns inline code into `\texttt{...}` with breaks allowed after `\_`, `/`, `:`,
  `=` and `,`, and lets any word longer than 28 characters break between a lowercase and an
  uppercase letter and after `/` or `:`. Without it 19 lines overflow, two of them off the page
  (the long XBRL element names).
- The status table's separator row has unequal dashes (`|-------|----------|--------|`) so the
  script column is wide enough: pandoc takes relative widths from the dashes.

## What a good build looked like

2026-10-08: 29 pages, no missing characters, 6 overfull boxes of at most 11 pt.
