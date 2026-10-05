"""Paper equations and cross-references as LaTeX macros, for documents other than V_Main.

Reads Drafts/Deposit Competition/V_Main.tex and V_Main.aux (it never writes either) and writes
paper_equations_macros.tex next to them. A document that \\input's that file (slides, notes) gets:

  \\papereq{eq:4}       the paper's equation, displayed, tagged with the paper's number
  \\papereq*{eq:4}      the same without the number
  \\papereqbody{eq:4}   the equation as one math box, for your own math environment
  \\ref / \\eqref       every label of V_Main resolves to the paper's number, e.g.
                       \\ref{estimation:single_idx} -> (III), so the generated tables, whose headers
                       and notes use those labels, render as they do in the paper. In beamer the
                       number is plain text (beamer would link it to a destination the deck lacks)
  \\paperlinkto{label}{target}  in beamer, link every reference to that paper label to a deck
                       target (a frame label or a \\hypertarget)
  the paper's own math commands (\\expectedvalue, \\indicator, ...), unless already defined

EQUATIONS. Every labelled row of a display environment (equation, align, gather, multline, flalign,
alignat). A label owns its row and the unlabelled rows before it, so a derivation that continues
over several lines stays whole; rows after the last label go to that label. Commented-out
environments are skipped. \\label, \\tag, \\nonumber and \\notag are dropped, and so is the sentence
punctuation that closes the equation (--keep-punct keeps it).

NUMBERS. Equation numbers and label texts come from V_Main.aux, so they are those of the last
compile of the paper. Recompile V_Main, then rerun this script, after the numbering changes.

Usage:
  python make_paper_equations.py
  python make_paper_equations.py --keep-punct
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import os
import pathlib
import re
import sys

from utils import paths as _paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DRAFTS = _paths.drafts_dir()
OUT_NAME = "paper_equations_macros.tex"

MATH_ENVS = ("equation", "align", "gather", "multline", "flalign", "alignat")
ALIGNED = {"align": "aligned", "flalign": "aligned", "alignat": "alignedat"}
MATH_PACKAGES = ("amssymb", "amsfonts", "mathtools", "mathrsfs", "dsfont", "relsize", "bm", "bbm",
                 "upgreek", "stmaryrd")

_COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*")
_ENV = re.compile(r"\\begin\{(%s)(\*?)\}" % "|".join(MATH_ENVS))
_LABEL = re.compile(r"\\label\s*\{([^}]*)\}")
_ROW_SKIP = re.compile(r"\[\s*-?[\d.]+\s*(?:pt|em|ex|mm|cm|in|bp|mu)\s*\]")
_DELIM_DOT = re.compile(r"\\(?:right|left|[bB]igg?[lrm]?)\s*\.$")


def strip_comments(text: str) -> str:
    return "\n".join(_COMMENT.sub(r"\1", ln) for ln in text.split("\n"))


def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def read_group(s: str, i: int):
    """The balanced {...} that opens at s[i]: (content, index after the closing brace)."""
    if i >= len(s) or s[i] != "{":
        raise ValueError("expected an opening brace")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def balanced(s: str) -> bool:
    depth, j = 0, 0
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        j += 1
    return depth == 0


# --------------------------------------------------------------------------------------------
# Preamble: math packages and the paper's own commands
# --------------------------------------------------------------------------------------------
def math_packages(pre: str):
    found = []
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", pre):
        for p in m.group(1).split(","):
            p = p.strip()
            if p in MATH_PACKAGES and p not in found:
                found.append(p)
    return found


def preamble_macros(pre: str):
    """\\newcommand, \\DeclareMathOperator and \\DeclarePairedDelimiter of the preamble, rewritten so
    that a document which already defines the name keeps its own definition."""
    out = []
    for m in re.finditer(r"\\newcommand\*?", pre):
        try:
            i = skip_ws(pre, m.end())
            if pre[i] == "{":
                name, i = read_group(pre, i)
                name = name.strip()
            else:
                mm = re.match(r"\\[A-Za-z@]+", pre[i:])
                if not mm:
                    continue
                name, i = mm.group(0), i + mm.end()
            opts = ""
            while True:
                j = skip_ws(pre, i)
                if j >= len(pre) or pre[j] != "[":
                    break
                k = pre.index("]", j)
                opts += pre[j:k + 1]
                i = k + 1
            body, i = read_group(pre, skip_ws(pre, i))
        except (ValueError, IndexError):
            continue
        out.append("\\providecommand{%s}%s{%s}" % (name, opts, body))
    for m in re.finditer(r"\\(DeclareMathOperator\*?|DeclarePairedDelimiter)\s*\{?\s*(\\[A-Za-z]+)\s*\}?",
                         pre):
        try:
            args, i = [], m.end()
            for _ in range(2 if m.group(1) == "DeclarePairedDelimiter" else 1):
                g, i = read_group(pre, skip_ws(pre, i))
                args.append("{%s}" % g)
        except (ValueError, IndexError):
            continue
        out.append("\\ifdefined%s\\else\\%s{%s}%s\\fi" % (m.group(2), m.group(1), m.group(2), "".join(args)))
    return out


# --------------------------------------------------------------------------------------------
# Equations
# --------------------------------------------------------------------------------------------
def split_rows(s: str):
    """Split a display body at the \\\\ that sit outside every brace group and inner environment."""
    rows, cur, depth, envs, i = [], [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if s.startswith("\\\\", i):
                if depth == 0 and envs == 0:
                    i += 2
                    if i < len(s) and s[i] == "*":
                        i += 1
                    mm = _ROW_SKIP.match(s, i)
                    if mm:
                        i = mm.end()
                    rows.append("".join(cur))
                    cur = []
                    continue
                cur.append("\\\\")
                i += 2
                continue
            if s.startswith("\\begin{", i):
                envs += 1
            elif s.startswith("\\end{", i):
                envs -= 1
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        cur.append(c)
        i += 1
    rows.append("".join(cur))
    return rows


def strip_tags(row: str) -> str:
    while True:
        m = re.search(r"\\tag\*?\s*(?=\{)", row)
        if not m:
            return row
        try:
            _, j = read_group(row, m.end())
        except ValueError:
            return row
        row = row[:m.start()] + row[j:]


def clean_row(row: str):
    labels = [x.strip() for x in _LABEL.findall(row)]
    row = _LABEL.sub("", row)
    row = strip_tags(row)
    row = re.sub(r"\\(?:nonumber|notag)(?![A-Za-z])", "", row)
    row = re.sub(r"\s+", " ", row).strip()
    row = re.sub(r"\s+([.,;])$", r"\1", row)
    return labels, row


def strip_punct(row: str) -> str:
    if row[-1:] in ".,;" and not _DELIM_DOT.search(row):
        return row[:-1].rstrip()
    return row


def equations(body: str, keep_punct: bool):
    """{label: the equation as one math box}, in the order of the paper."""
    out, skipped = {}, []
    for m in _ENV.finditer(body):
        env = m.group(1) + m.group(2)
        end = body.find("\\end{%s}" % env, m.end())
        if end < 0:
            continue
        inner = body[m.end():end]
        if "\\label" not in inner:
            continue
        arg = ""
        if m.group(1) == "alignat":
            try:
                g, i = read_group(inner, skip_ws(inner, 0))
            except ValueError:
                continue
            arg, inner = "{%s}" % g, inner[i:]
        groups, pending, last = [], [], None
        for row in split_rows(inner):
            labels, txt = clean_row(row)
            if not txt and not labels:
                continue
            pending.append(txt)
            if labels:
                last = (labels, pending)
                groups.append(last)
                pending = []
        if pending and last is not None:
            last[1].extend(pending)
        for labels, rows in groups:
            rows = [r for r in rows if r]
            if not rows:
                continue
            if not keep_punct:
                rows[-1] = strip_punct(rows[-1])
            if m.group(1) in ALIGNED:
                w = ALIGNED[m.group(1)]
                box = "\\begin{%s}%s %s \\end{%s}" % (w, arg, " \\\\ ".join(rows), w)
            elif len(rows) > 1:
                box = "\\begin{gathered} %s \\end{gathered}" % " \\\\ ".join(rows)
            else:
                box = rows[0]
            for lab in labels:
                if balanced(box) and "#" not in box:
                    out[lab] = box
                else:
                    skipped.append(lab)
    return out, skipped


# --------------------------------------------------------------------------------------------
# Labels of the compiled paper
# --------------------------------------------------------------------------------------------
def aux_labels(aux: str):
    """{label: (text, page, anchor)} from the \\newlabel lines of V_Main.aux."""
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}", aux):
        key = m.group(1)
        if "@" in key:
            continue
        try:
            payload, _ = read_group(aux, skip_ws(aux, m.end()))
            parts, j = [], 0
            while True:
                j = skip_ws(payload, j)
                if j >= len(payload) or payload[j] != "{":
                    break
                g, j = read_group(payload, j)
                parts.append(g)
        except (ValueError, IndexError):
            continue
        if len(parts) >= 2 and "#" not in parts[0]:
            out[key] = (parts[0], parts[1], parts[3] if len(parts) > 3 else "")
    return out


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
TEX_HEADER = r"""% paper_equations_macros.tex -- GENERATED by make_paper_equations.py. Do not edit.
% Generated __STAMP__ from __MAIN__ and its .aux. Regenerate: python make_paper_equations.py
% For documents OTHER than the paper (slides, notes): \input this file in the preamble, after
% hyperref when hyperref is used. The paper must not input it: its labels would be defined twice.
%
% \papereq{eq:4}       the paper's equation, displayed, tagged with the paper's number
% \papereq*{eq:4}      the same without the number
% \papereqbody{eq:4}   the equation as one math box, for your own math environment or a \resizebox
% \ref{..}, \eqref{..} every label of the paper resolves to the paper's number, e.g.
%                      \ref{estimation:single_idx} -> (III).
% \paperlinkto{estimation:single_idx}{frame:routines}
%                      in beamer: make every reference to that paper label a link to the deck
%                      target named in the second argument (a frame label or a \hypertarget).
% Links. In beamer a reference to a paper label is plain text: beamer's own \ref links to a
% destination named after the label, which a deck does not have, so the link would jump to a
% wrong page. \paperlinkto opts one label back in, pointed at a slide. Outside beamer, with
% hyperref, the reference links into the paper's PDF; define \PEQpaperpdf before \input as the
% path to that PDF.
% The paper's own math commands (\expectedvalue, \indicator, ...) are provided below unless the
% document already defines them. An unknown equation label prints a red ?? with a LaTeX warning.
% Numbers are those of the last compile of the paper.
"""

TEX_DEFS = r"""\providecommand{\PEQpaperpdf}{__PDF__}
\providecommand{\PEQwarn}[1]{\mbox{\bfseries\color{red}??}\PackageWarning{paper-equations}{No such paper equation: #1}}
\providecommand{\PEQd}[3]{\expandafter\def\csname peq@#1@body\endcsname{#2}\expandafter\def\csname peq@#1@num\endcsname{#3}}
% The paper's labels are set at \begin{document}, after the document's own .aux is read. A paper
% table that is \input in the document carries its \label with it; set this late, the paper's
% number wins over the one that \label would give (in beamer, where captions are unnumbered, an
% empty one), and nothing is reported as multiply defined.
\ifdefined\hypersetup
\providecommand{\PEQsetlabel}[4]{\expandafter\gdef\csname r@#1\endcsname{{#2}{#3}{}{#4}{\PEQpaperpdf}}}
\else
\providecommand{\PEQsetlabel}[4]{\expandafter\gdef\csname r@#1\endcsname{{#2}{#3}}}
\fi
\providecommand{\PEQlabel}[4]{\expandafter\gdef\csname peq@lab@#1\endcsname{}\AtBeginDocument{\PEQsetlabel{#1}{#2}{#3}{#4}}}
\providecommand{\paperlinkto}[2]{\expandafter\gdef\csname peq@tgt@#1\endcsname{#2}}
% beamer redefines \ref at \begin{document} as \hyperlink{<label>}{<number>}. For a paper label
% that destination does not exist in the deck, so the reference is set as text, or linked to
% the target \paperlinkto names. Every other label keeps beamer's behaviour.
\edef\PEQrestoreat{\catcode`\noexpand\@=\the\catcode`\@\relax}
\makeatletter
\long\def\PEQ@ref#1{%
  \ifcsname peq@tgt@#1\endcsname
    \edef\PEQ@target{\csname peq@tgt@#1\endcsname}%
    \def\PEQ@next{\expandafter\hyperlink\expandafter{\PEQ@target}{\beamer@origref{#1}}}%
  \else\ifcsname peq@lab@#1\endcsname
    \def\PEQ@next{\beamer@origref{#1}}%
  \else
    \def\PEQ@next{\PEQ@beamerref{#1}}%
  \fi\fi
  \PEQ@next}
\AtBeginDocument{\ifdefined\beamer@ref\let\PEQ@beamerref\beamer@ref\let\beamer@ref\PEQ@ref\fi}
% At \end{document} LaTeX compares every label of the .aux with its current value. A paper label
% the document also defines would differ on every run, so it is left out of that comparison.
\def\PEQ@testdefpaper#1#2#3{\ifcsname peq@lab@#2\endcsname\else\PEQ@testdef{#1}{#2}{#3}\fi}
\AtBeginDocument{\let\PEQ@testdef\@testdef\let\@testdef\PEQ@testdefpaper}
\PEQrestoreat
\providecommand{\papereqbody}[1]{\ifcsname peq@#1@body\endcsname\csname peq@#1@body\endcsname\else\PEQwarn{#1}\fi}
\NewDocumentCommand{\papereq}{s m}{%
  \ifcsname peq@#2@body\endcsname
    \begin{equation*}\csname peq@#2@body\endcsname
    \IfBooleanF{#1}{\expandafter\ifx\csname peq@#2@num\endcsname\empty\else\tag{\csname peq@#2@num\endcsname}\fi}%
    \end{equation*}%
  \else\PEQwarn{#2}\fi}
"""


def render(main: pathlib.Path, packages, macros, eqs, labels) -> str:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [TEX_HEADER.replace("__STAMP__", stamp).replace("__MAIN__", main.name).rstrip("\n")]
    lines.append("\\RequirePackage{amsmath}")
    lines += ["\\RequirePackage{%s}" % p for p in packages]
    lines.append(TEX_DEFS.replace("__PDF__", main.with_suffix(".pdf").name).rstrip("\n"))
    lines.append("")
    lines.append("% --- the paper's math commands ---")
    lines += macros
    lines.append("")
    lines.append("% --- equations ---")
    for lab, box in eqs.items():
        lines.append("\\PEQd{%s}{%s}{%s}" % (lab, box, labels.get(lab, ("",))[0]))
    lines.append("")
    lines.append("% --- labels of the paper ---")
    for key, (text, page, anchor) in labels.items():
        lines.append("\\PEQlabel{%s}{%s}{%s}{%s}" % (key, text, page, anchor))
    return "\n".join(lines) + "\n"


def _dump(path: pathlib.Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", type=pathlib.Path, default=DRAFTS / "V_Main.tex",
                    help="the paper's main .tex (read only); its .aux is read from the same folder")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help=f"output file (default: {OUT_NAME} next to the paper)")
    ap.add_argument("--keep-punct", action="store_true",
                    help="keep the sentence punctuation that closes an equation")
    args = ap.parse_args()

    src = args.main.read_text(encoding="utf-8")
    pre, sep, body = src.partition("\\begin{document}")
    if not sep:
        sys.exit(f"{args.main}: no \\begin{{document}}")
    pre = strip_comments(pre)
    body = strip_comments(body.partition("\\end{document}")[0])

    aux = args.main.with_suffix(".aux")
    labels = aux_labels(aux.read_text(encoding="utf-8", errors="replace")) if aux.exists() else {}
    if not labels:
        print(f"  WARNING: no labels read from {aux}; equations carry no number and \\ref will not "
              "resolve. Compile the paper, then rerun.")

    eqs, skipped = equations(body, args.keep_punct)
    out = args.out or args.main.with_name(OUT_NAME)
    _dump(out, render(args.main, math_packages(pre), preamble_macros(pre), eqs, labels))

    unnumbered = [k for k in eqs if k not in labels]
    print(f"  {len(eqs)} equations, {len(labels)} labels")
    if unnumbered:
        print("  no number in the .aux for: " + ", ".join(unnumbered))
    if skipped:
        print("  skipped (unbalanced braces or a # in the body): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
