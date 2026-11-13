r"""make_state_vector_table.py -- the appendix table that documents the sleepiness state vector.

Author: Pedro Feijó de Moraes

Renders tab_state_vector_map.tex (label tab:state_vector_map), one table float, `table[htpb]`, in
two panels:

  Panel A  every state of the specification (12) block: the depositor trait it measures, the
           share of its variance the entity and quarter fixed effects absorb, and its law of
           motion in the forward simulation. Four columns, laid on the five-column grid of Panel
           B: the law of motion spans the last two.
  Panel B  the conditioning of the two-way-demeaned regressors of eq. (13) when a second moment
           of a characteristic already in the block is added: the collected candidates the block
           leaves out (GDP per capita, the young share, the 4G/5G share of mobile lines).

Every number comes from ESTIMATION_OUTPUT/DIAG_STATE_SELECTION, written by
diagnostics/diag_state_vector_selection.py from the regressors alone. The unrounded values go to
the sidecar tab_state_vector_map.json, which make_paper_numbers.py reads. The sidecar also keeps
the absorbed share of the 4G/5G share, a collected candidate the table does not print, so that the
prose can state why it is left out; its field `not_in_table` lists such states.

    python make_state_vector_table.py
"""
from __future__ import annotations

import json
import sys

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pandas as pd

from utils import paths as _paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIAG = _paths.diag_dir("STATE_SELECTION")
BASENAME = "tab_state_vector_map"
LABEL = "tab:state_vector_map"
PROD_BLOCK = "S5"

# Panel A. (state, label, trait, block whose design row carries the state, law of motion).
# Labels follow sleep_export_link._BASE_LABELS; the laws of motion are those of
# bbl_sleep_link.ROLE_OF and bbl_transitions.S_COLS.
IN_ROWS = [
    ("risk_free_qoq_lag", "Lagged Selic Rate", "Common: stakes of shopping",
     PROD_BLOCK, "Focus forward curve, lagged one quarter"),
    ("pix_exists", "Pix Available", "Common: cost of moving funds",
     PROD_BLOCK, "Held at the launch quarter"),
    ("cadunico_families_per1000", r"Cad\'{U}nico Families", "Composition: income",
     PROD_BLOCK, "AR(1) within market"),
    ("fraction_65plus", "Fraction 65+", "Composition: age",
     PROD_BLOCK, "AR(1) within market"),
    ("connections_per100", "Mobile Lines", "Composition: digital access",
     PROD_BLOCK, "AR(1) within market"),
]

# Collected candidates the table does not print, whose absorbed share the sidecar keeps for the
# prose. (state, block whose design row carries the state)
SIDECAR_ONLY = [
    ("frac_4g5g", "S5+frac_4g5g"),
]

# Panel B. (block, row label, what it adds).
DESIGN_ROWS = [
    (PROD_BLOCK, "Specification (12)", "--"),
    ("S5+gdp_per_capita", r"+ GDP \textit{per capita}", "Second income moment"),
    ("S5+fraction_young", "+ Fraction Young", "Second age moment"),
    ("S7", "+ both", "Second income and age moments"),
    ("S5+frac_4g5g", "+ 4G/5G Share", "Second digital-access moment"),
]


# Panel A has four columns on the five-column grid of the table: the absorbed share sits in the
# third (centred, as wide as it) and the law of motion spans the fourth and fifth.
_ABS_W = r"3.1cm"
_MOTION_W = r"\dimexpr 3.5cm+1.8cm+2\tabcolsep\relax"


def _abs(cell: str) -> str:
    return (r"\multicolumn{1}{>{\centering\arraybackslash}p{" + _ABS_W + "}}{" + cell + "}")


def _motion(cell: str) -> str:
    return (r"\multicolumn{2}{>{\raggedright\arraybackslash}p{" + _MOTION_W + "}}{" + cell + "}")


def f3(x) -> str:
    """Three decimals, never -0.000."""
    s = f"{float(x):.3f}"
    return "0.000" if s == "-0.000" else s


def f1c(x) -> str:
    """A design statistic at three decimals with thousands separators."""
    return f"{float(x):,.3f}"


def load():
    design = pd.read_csv(DIAG / "design_by_block.csv").set_index("block")
    per_var = pd.read_csv(DIAG / "design_by_variable.csv")
    meta = json.loads((DIAG / "meta.json").read_text(encoding="utf-8"))
    return design, per_var, meta


def absorbed(per_var: pd.DataFrame, block: str, state: str) -> float:
    r = per_var[(per_var["block"] == block) & (per_var["state"] == state)]
    if len(r) != 1:
        raise KeyError(f"design_by_variable.csv has no row for {state} in block {block}")
    return float(r["absorbed_alone"].iloc[0])


def _float_table(body, col_fmt, caption, label, header, footnote, ncols) -> str:
    r"""One \input-ready table as a `table[htpb]` float: the caption and label above a full-width
    tabularx, the notes as its last row. The table fits on one page, so it floats whole; the
    spacing, font size and column padding are set inside the float and do not leak out of it."""
    return "\n".join([
        r"\begin{table}[htpb]",
        r"\setstretch{1.0}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_fmt}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        rf"\multicolumn{{{ncols}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {footnote}}} \\",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def render(design, per_var) -> tuple[str, dict]:
    side = {"absorbed_alone": {}, "not_in_table": [], "design": {}}
    body = [r"\multicolumn{5}{@{}l}{\textit{Panel A. States of the sleepiness function}} \\*",
            r"\addlinespace[0.2em]"]
    for state, lab, trait, blk, motion in IN_ROWS:
        a = absorbed(per_var, blk, state)
        side["absorbed_alone"][state] = a
        body.append(f"{lab} & {trait} & {_abs(f3(a))} & {_motion(motion)} \\\\")
    for state, blk in SIDECAR_ONLY:
        side["absorbed_alone"][state] = absorbed(per_var, blk, state)
        side["not_in_table"].append(state)
    body += [r"\midrule",
             r"\multicolumn{5}{@{}l}{\textit{Panel B. Conditioning of the regressors of the second "
             r"stage of the sleepiness estimation}} \\*",
             r"\addlinespace[0.2em]",
             r"Block & Adds & \multicolumn{1}{c}{Condition number} & Effective dimension & "
             r"\multicolumn{1}{c}{Largest VIF} \\*",
             r"\addlinespace[0.1em]"]
    for blk, lab, adds in DESIGN_ROWS:
        if blk not in design.index:
            continue
        d = design.loc[blk]
        side["design"][blk] = dict(cond=float(d["cond"]),
                                   effdim=float(d["effdim_participation"]),
                                   max_vif=float(d["max_vif"]), max_vif_col=str(d["max_vif_col"]),
                                   n_cols=int(d["n_cols"]))
        body.append(f"{lab} & {adds} & \\multicolumn{{1}}{{c}}{{{f1c(d['cond'])}}} & "
                    f"{f3(d['effdim_participation'])} & "
                    f"\\multicolumn{{1}}{{c}}{{{f1c(d['max_vif'])}}} \\\\")
    note = (
        r"\textit{Notes:} Panel A: the states at specification (12). Absorbed by FE: the share "
        r"of the state's variance absorbed by bank $\times$ type $\times$ market and quarter "
        r"fixed effects, on the specification (12) sample. "
        r"Panel B: the two-way-demeaned regressors of the second stage of the sleepiness "
        r"estimation. VIF: variance inflation factor. Definitions: Appendix "
        r"\ref{app:empirical:sleep}."
    )
    head = (r"\textbf{Variable} & \textbf{Depositor trait} & "
            + _abs(r"\textbf{Absorbed by FE}") + " & "
            + _motion(r"\textbf{Law of motion}"))
    col_fmt = (r">{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}X "
               r">{\raggedright\arraybackslash}p{3.1cm} >{\centering\arraybackslash}p{1.8cm} "
               r">{\raggedright\arraybackslash}p{3.5cm}")
    tex = _float_table(body, col_fmt, "State Vector of the Sleepiness Function", LABEL, head,
                       note, ncols=5)
    return tex, side


def main() -> int:
    design, per_var, meta = load()
    tex, side = render(design, per_var)
    side["source"] = str(DIAG)
    side["diag_written_utc"] = meta.get("written_utc")
    side["gates"] = meta.get("gates")
    for d in (_paths.rout_dir(), _paths.drafts_dir()):
        (d / f"{BASENAME}.tex").write_text(tex, encoding="utf-8")
        print(f"  Wrote {d / (BASENAME + '.tex')}")
    sc = _paths.rout_dir() / f"{BASENAME}.json"
    sc.write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(f"  Wrote {sc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
