"""Copy the four variance tables from Rout/slide_anova_tables.md into the professor-facing note.

Each block is a "Table:" caption line, a blank line, and the pipe-table rows. A block replaces the
block in the note whose caption line matches its first 60 characters. Run after
`make_slide_figures.py --only anova`, then rebuild the PDF (build.sh).
"""
import sys
from pathlib import Path

OPEN_FINANCE = Path(__file__).resolve().parents[4]
FRAGMENT = (OPEN_FINANCE / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT"
            / "Rout" / "slide_anova_tables.md")
NOTE = OPEN_FINANCE / "Drafts" / "Deposit Competition" / "motivation_figures_review.md"


def blocks(lines):
    """Start and end (exclusive) of each caption + table block."""
    out = []
    for i, line in enumerate(lines):
        if line.startswith("Table:"):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append((i, j))
    return out


frag = FRAGMENT.read_text(encoding="utf-8").split("\n")
note = NOTE.read_text(encoding="utf-8").split("\n")
new = {frag[i][:60]: frag[i:j] for i, j in blocks(frag)}
spliced = 0
for i, j in reversed(blocks(note)):
    key = note[i][:60]
    if key in new:
        note[i:j] = new.pop(key)
        spliced += 1
if new:
    sys.exit(f"captions not found in the note: {list(new)}")
NOTE.write_text("\n".join(note), encoding="utf-8", newline="")
print("spliced", spliced)
