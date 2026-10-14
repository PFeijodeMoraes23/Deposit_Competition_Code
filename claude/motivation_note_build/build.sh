#!/usr/bin/env bash
# Build motivation_figures_review.pdf from its md (Git Bash; pandoc + MiKTeX xelatex on PATH).
# Usage: bash build.sh [build_dir]. The build dir defaults to a folder under $TEMP, outside
# OneDrive. Figure PDFs are copied there from the note's folder, Drafts/Deposit Competition
# (prof.lua swaps each PNG in the md for its PDF twin by bare name). Stops if xelatex reports
# an overfull line.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OF="$(cd "$HERE/../../../.." && pwd)"
NOTE_DIR="$OF/Drafts/Deposit Competition"
FIG_DIR="$NOTE_DIR"
B="${1:-${TEMP:-/tmp}/motivation_note_build}"
mkdir -p "$B"

grep -o '](<[^>]*\.png>)' "$NOTE_DIR/motivation_figures_review.md" \
  | sed 's/^](<//; s/\.png>)$/.pdf/' | sort -u \
  | while read -r f; do cp "$FIG_DIR/$f" "$B/"; done

cd "$NOTE_DIR"
pandoc motivation_figures_review.md -o "$B/note.tex" \
  --from markdown-implicit_figures-raw_tex --lua-filter "$HERE/prof.lua" \
  --standalone --number-sections --toc --toc-depth=2 -H "$HERE/prof_header.tex" \
  -V documentclass=article -V geometry:margin=2.5cm -V fontsize=11pt -V linestretch=1.15 \
  -V colorlinks=true -V linkcolor=black -V urlcolor=blue \
  -V "mainfont=Palatino Linotype" -V monofont=Consolas --columns=100

cd "$B"
for i in 1 2 3; do xelatex -interaction=nonstopmode note.tex > "x$i.out" 2>&1 || true; done
grep -q "Output written on note.pdf" x3.out || { echo "xelatex failed, see $B/x3.out"; exit 1; }
n=$(grep -c "Overfull" note.log || true)
if [ "$n" != "0" ]; then echo "$n overfull lines, see $B/note.log"; exit 1; fi
cp note.pdf "$NOTE_DIR/motivation_figures_review.pdf"
grep "Output written" x3.out
