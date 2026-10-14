"""make_notes_pdf.py -- render a notes markdown to PDF with the house builder, three xelatex
passes, and long code paths allowed to break.

The builder is `Drafts/Deposit Competition/build_notes_pdf.py`, used unchanged. This script
only redirects its pandoc call:
- pandoc writes a standalone .tex into a short build folder;
- a header lets `\\texttt` paths break at underscores and slashes;
- xelatex runs three times, so the table of contents settles;
- the PDF is copied next to the markdown.

usage: python make_notes_pdf.py <notes .md> [--report] [--no-break-paths] [--build-dir DIR]

The .md must sit in the folder that holds build_notes_pdf.py (Drafts/Deposit Competition).
`--report` selects the builder's report layout (numbered sections and a title block). The build
folder defaults to %TEMP%\\npdf. Keep it short: past Windows' 260-character path limit xelatex
writes no .log and the build fails without saying why.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Lets long code paths break at underscores and slashes. \texttt is redefined to insert a
# break point after every "/" of its argument; \_ gets one after every underscore.
BREAK_PATHS_TEX = r"""% make_notes_pdf.py: long code paths may break at underscores and slashes.
% A paragraph with long unbreakable words (citation keys, URLs) gets looser spacing instead of
% running into the margin; ordinary paragraphs are unaffected.
\tolerance=2000
\setlength{\emergencystretch}{6em}
\let\npOrigUnderscore\_
\renewcommand{\_}{\npOrigUnderscore\allowbreak}
\let\npOrigTexttt\texttt
\renewcommand{\texttt}[1]{\npOrigTexttt{\npSlashBreak#1\npEnd}}
\makeatletter
\def\npSlashBreak#1\npEnd{\npSlashLoop#1/\npStop}
\def\npSlashLoop#1/#2\npStop{#1\ifx\relax#2\relax\else/\allowbreak\npSlashLoop#2\npStop\fi}
\makeatother
"""


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    md = Path(args[0]).resolve()
    rest = args[1:]
    break_paths = "--no-break-paths" not in rest
    rest = [a for a in rest if a != "--no-break-paths"]
    build = Path(tempfile.gettempdir()) / "npdf"
    if "--build-dir" in rest:
        i = rest.index("--build-dir")
        build = Path(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    folder = md.parent
    if not (folder / "build_notes_pdf.py").exists():
        raise SystemExit(f"{folder} has no build_notes_pdf.py; the notes must sit next to the builder")
    build.mkdir(parents=True, exist_ok=True)
    header = build / "break_paths.tex"
    header.write_text(BREAK_PATHS_TEX, encoding="utf-8")

    sys.path.insert(0, str(folder))
    import build_notes_pdf as b  # noqa: E402  (the house builder, unchanged)

    target_pdf = md.with_suffix(".pdf")
    tex = build / (md.stem + ".tex")
    real_run = subprocess.run

    def patched_run(cmd, *a, **k):
        if not (cmd and cmd[0] == "pandoc"):
            return real_run(cmd, *a, **k)
        i = cmd.index("-o")
        cmd = cmd[:i] + ["-o", str(tex), "--standalone"] + cmd[i + 2:]
        cmd = [c for c in cmd if not c.startswith("--pdf-engine")]
        if break_paths:
            cmd = cmd + ["--include-in-header", str(header)]
        res = real_run(cmd, *a, **k)
        if res.returncode != 0:
            return res
        for _ in range(3):
            res = real_run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=build, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
            if res.returncode != 0:
                print(res.stdout[-3000:])
                return res
        log = (build / (tex.stem + ".log")).read_text(encoding="utf-8", errors="replace")
        n_over = sum(1 for ln in log.splitlines() if ln.startswith("Overfull \\hbox"))
        print(f"rerun warning left: {'Rerun to get' in log}; overfull lines: {n_over}")
        shutil.copyfile(build / (tex.stem + ".pdf"), target_pdf)
        return res

    b.subprocess.run = patched_run
    os.chdir(folder)
    sys.argv = ["build_notes_pdf.py", md.name] + rest
    return b.main()


if __name__ == "__main__":
    raise SystemExit(main())
