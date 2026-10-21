r"""make_state_vector_table.py -- the appendix table that documents the sleepiness state vector.

Author: Pedro Feijó de Moraes

Renders tab_state_vector_map.tex (label tab:state_vector_map), one non-breaking table in two
panels:

  Panel A  every state of the specification (12) block and every collected candidate left out:
           the depositor trait it measures, its analogue in the county-level sleepiness
           specification of Egan et al. (2025), the share of its variance the entity and quarter
           fixed effects absorb, and its law of motion in the forward simulation (or the reason it
           is left out).
  Panel B  the conditioning of the two-way-demeaned regressors of eq. (13) when a second moment
           of a characteristic already in the block is added.

Every number comes from ESTIMATION_OUTPUT/DIAG_STATE_SELECTION, written by
diagnostics/diag_state_vector_selection.py from the regressors alone. The unrounded values go to
the sidecar tab_state_vector_map.json, which make_paper_numbers.py reads.

    python make_state_vector_table.py
"""
from __future__ import annotations

import json
import sys

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pandas as pd

from make_bbl_cost_tables import _wrap
from utils import paths as _paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIAG = _paths.diag_dir("STATE_SELECTION")
BASENAME = "tab_state_vector_map"
LABEL = "tab:state_vector_map"
PROD_BLOCK = "S5"

# Panel A. (state, label, trait, Egan et al. analogue, block whose design row carries the state,
# law of motion or reason left out). Labels follow sleep_export_link._BASE_LABELS; the laws of
# motion are those of bbl_sleep_link.ROLE_OF and bbl_transitions.S_COLS.
IN_ROWS = [
    ("risk_free_qoq_lag", "Lagged Selic Rate", "Common: stakes of shopping",
     "Lagged federal funds rate", PROD_BLOCK, "Focus forward curve, lagged one quarter"),
    ("pix_exists", "Pix Available", "Common: cost of moving funds",
     "--", PROD_BLOCK, "Held at the launch quarter"),
    ("cadunico_families_per1000", r"Cad\'{U}nico Families", "Composition: income",
     r"Share with income above \$50k", PROD_BLOCK, "AR(1) within market"),
    ("fraction_65plus", "Fraction 65+", "Composition: age",
     "Share aged over 65", PROD_BLOCK, "AR(1) within market"),
    ("connections_per100", "Mobile Lines", "Composition: digital access",
     "Share with internet access", PROD_BLOCK, "AR(1) within market"),
]
OUT_ROWS = [
    ("gdp_per_capita", r"GDP \textit{per capita}", "Composition: income",
     "--", "S7", "Second income moment; enters demand ($\\pi$)"),
    ("fraction_young", "Fraction Young (15--20)", "Composition: age",
     "--", "S7", "Second age moment"),
    ("frac_4g5g", "4G/5G Share of Mobile Lines", "Composition: digital access",
     "--", "S5+frac_4g5g", "Second digital-access moment"),
    (None, "Pix Users and Transactions", "Pix usage intensity",
     "--", None, "Equilibrium object: adoption responds to attention"),
    (None, "Branch and Access-Point Density", "Institutions' networks",
     "--", None, "Chosen by the institutions"),
]

# Panel B. (block, row label, what it adds).
DESIGN_ROWS = [
    (PROD_BLOCK, "Specification (12)", "--"),
    ("S5+gdp_per_capita", r"+ GDP \textit{per capita}", "Second income moment"),
    ("S5+fraction_young", "+ Fraction Young", "Second age moment"),
    ("S7", "+ both", "Second income and age moments"),
    ("S5+frac_4g5g", "+ 4G/5G Share", "Second digital-access moment"),
]


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


def render(design, per_var) -> tuple[str, dict]:
    side = {"absorbed_alone": {}, "design": {}}
    body = [r"\multicolumn{5}{@{}l}{\textit{Panel A. States of the sleepiness function}} \\*",
            r"\addlinespace[0.2em]",
            r"\multicolumn{5}{@{}l}{In the state vector} \\*"]
    for state, lab, trait, analogue, blk, motion in IN_ROWS:
        a = absorbed(per_var, blk, state)
        side["absorbed_alone"][state] = a
        body.append(f"{lab} & {trait} & {analogue} & {f3(a)} & {motion} \\\\")
    body += [r"\addlinespace[0.3em]",
             r"\multicolumn{5}{@{}l}{Collected, not in the state vector} \\*"]
    for state, lab, trait, analogue, blk, reason in OUT_ROWS:
        if state is not None and blk in set(per_var["block"]):
            a = absorbed(per_var, blk, state)
            side["absorbed_alone"][state] = a
            cell = f3(a)
        else:
            cell = "--"
        body.append(f"{lab} & {trait} & {analogue} & {cell} & {reason} \\\\")
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
        r"\textit{Notes:} Panel A: the states at specification (12) and the collected candidates "
        r"left out. Common states move the stakes of shopping or the cost of moving funds for "
        r"every depositor. Each composition state is the market share of one depositor "
        r"characteristic. Analogue: the common state, or the county-level variable to which "
        r"\textcite{egan2025dynamic} map the depositor characteristic, in their sleepiness "
        r"function. Absorbed by FE: one minus the ratio of the state's variance net of bank "
        r"$\times$ type $\times$ market and quarter fixed effects to its raw variance, on the "
        r"specification (12) sample. A national state is identified only through its product with "
        r"lagged deposits. Laws of motion: first-order autoregressions within market, on "
        r"consecutive quarters of the market panel, for the composition states, the Focus forward "
        r"curve lagged one quarter for the Selic rate, and Pix held at its launch value. Panel B: "
        r"the two-way-demeaned regressors of the second stage of the sleepiness estimation, "
        r"lagged deposits and the "
        r"control-function term included. Condition number: square root of the ratio of the "
        r"extreme eigenvalues of their correlation matrix. Effective dimension: "
        r"$(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ over those eigenvalues. VIF: variance "
        r"inflation factor. Both panels use the regressors only, not the deposit outcome."
    )
    head = (r"\textbf{Variable} & \textbf{Depositor trait} & \textbf{Analogue in Egan et al.} & "
            r"\textbf{Absorbed by FE} & \textbf{Law of motion, or reason left out}")
    col_fmt = (r">{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}X "
               r">{\raggedright\arraybackslash}p{3.1cm} >{\centering\arraybackslash}p{1.8cm} "
               r">{\raggedright\arraybackslash}p{3.5cm}")
    tex = _wrap(body, col_fmt, "State Vector of the Sleepiness Function", LABEL, head, note,
                ncols=5, nobreak=True)
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
