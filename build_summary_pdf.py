#!/usr/bin/env python3
"""
build_summary_pdf.py — compile Drafts/Deposit Competition/SUMMARY.md to SUMMARY.pdf.

PORTRAIT document, with the wide "Logit vs RC-BLP stages" compare tables placed on
LANDSCAPE pages (pdflscape). The source SUMMARY.md stays pure markdown — the landscape
wrapping and the folding of the few unicode glyphs standard Windows text fonts lack happen
here, on a temp copy. Requires pandoc + xelatex (Cambria/Consolas system fonts).

Run after process_blp_outputs.py (which regenerates SUMMARY.md):
    python build_summary_pdf.py
"""
import os, re, subprocess, sys, tempfile

from utils import paths as _paths

DRAFT = str(_paths.drafts_dir())
SRC = os.path.join(DRAFT, "SUMMARY.md")
PDF = os.path.join(DRAFT, "SUMMARY.pdf")
# References.bib lives one level up (Drafts/); pandoc --citeproc resolves the [@key] citations.
BIB = os.path.join(os.path.dirname(DRAFT), "References.bib")
# Glyphs that Latin/Windows text fonts lack → folded for the PDF only (the .md keeps unicode).
# The combining marks (circumflex δ̂, macron D̄, tilde D̃) can't sit on a Latin base letter, so
# they fold to nothing (the bar/hat is dropped in the PDF only).
FOLD = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '̂': '', '̄': '', '̃': '', '→': '->', '−': '-',
        '∈': ' in ', '≥': '>=', '≤': '<=', '≈': '~=', '∅': '(none)'}
# Wide tables that should sit on their own landscape pages (each from its H2 header to the next H2).
LANDSCAPE_SECTIONS = ("## Weak-instruments diagnostics", "## Logit vs RC-BLP stages")
# \blandscape/\elandscape are COMMANDS (not the landscape ENVIRONMENT) so pandoc still parses
# the markdown tables between them instead of swallowing them as raw LaTeX.
HEADER_TEX = (r"\usepackage{pdflscape}" "\n"
              r"\newcommand{\blandscape}{\begin{landscape}}" "\n"
              r"\newcommand{\elandscape}{\end{landscape}}" "\n")

def main():
    if not os.path.exists(SRC):
        sys.exit(f"ERROR: {SRC} not found — run process_blp_outputs.py first.")
    t = open(SRC, encoding="utf-8").read()
    for k, v in FOLD.items():
        t = t.replace(k, v)
    lines, out, i = t.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]
        if any(ln.startswith(s) for s in LANDSCAPE_SECTIONS):
            out += ["", r"\blandscape", r"\footnotesize", ""]
            out.append(ln); i += 1
            while i < len(lines) and not lines[i].startswith("## "):   # to the next H2 (Notes)
                out.append(lines[i]); i += 1
            out += ["", r"\normalsize", r"\elandscape", ""]
        else:
            out.append(ln); i += 1
    tmp = tempfile.gettempdir()
    md, hdr = os.path.join(tmp, "_summary_pdf.md"), os.path.join(tmp, "_summary_header.tex")
    open(md, "w", encoding="utf-8").write("\n".join(out))
    open(hdr, "w", encoding="utf-8").write(HEADER_TEX)
    cmd = ["pandoc", md, "-o", PDF, "--pdf-engine=xelatex",
           "-V", "mainfont=Cambria", "-V", "monofont=Consolas",
           "-V", "geometry:margin=1in", "-H", hdr]
    if os.path.exists(BIB):                       # resolve [@key] citations + render bibliography
        cmd += ["--citeproc", "--bibliography=" + BIB]
    else:
        print(f"NOTE: {BIB} not found — citations will render unresolved.")
    r = subprocess.run(cmd, capture_output=True, text=True)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if re.search(r"error|! |cannot|fatal", l, re.I)]
    if errs:
        print("PDF build issues:\n" + "\n".join(errs[:12]))
    print("wrote", PDF if (r.returncode == 0 and os.path.exists(PDF)) else "(FAILED)")
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
