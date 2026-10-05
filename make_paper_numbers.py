r"""
make_paper_numbers.py
=====================
Every number the paper's prose quotes from the BBL cost estimation, the sleepy-share path of the
forward simulation and the demand elasticities, produced by a script. The prose placeholders of the
revision plan are filled from ESTIMATION_OUTPUT/PAPER_NUMBERS/paper_numbers.{json,md}, and each
number there is read from the output of the generator that renders the exhibit it belongs to:

  A  BBL (Tables 7-11, appendix up/down)   COST_FWD/bbl_table_numbers{psi_tag}.json, written by
                                           make_bbl_cost_tables.py from the objects it renders the
                                           tables from (default pass: T7, T9, T11; --from-psi: T8,
                                           T10, up/down). One file per run tag; the multi-start run
                                           the plan targets is --bbl-target (_ms982), the tagged
                                           single-curve run of T11 is --sc-target (_sc982).
  B  phi along the simulated path          COST_FWD/phi_path_summary_E{k}_spec_12.json
                                           (bbl_phi_path_summary.jl): pooled over launch quarters,
                                           pop weighting (headline), h = 0, 40, 250.
  C  on-impact market elasticities         (Mean own-price elasticity) x (1 - Mean phi-hat):
       logit elasticity (Table 5)          BLP_RESULTS/logit/logit_summary_spec_12.json, entry
                                           E{k}_full alpha, times mean rho(1-s) of the demand parquet
                                           (blp_logit.jl write_logit_comparison_table)
       BLP elasticity (Table 6)            BLP_RESULTS/cluster_raw/blp_results_E{k}_spec_12_ext1.json
                                           alpha, times the same mean (make_blp_rc_table
                                           semi_elast_cell, the function Table 6 is rendered with)
       Mean phi-hat (Table 4)              DEMAND_PREP/est{k}/national_phi_t.csv, the mean of
                                           phi_t_<SPEC12_TAG> (sleep_export_spec12_compare.py)
  D  entry-path interval                   DIAG_PHI_SEPARATION/d6_phi_in_interval.csv (lo, hi) and
                                           d6_implied_phi.csv (phi_entry of the B events)
  E  franchise-value counterfactual        dropped from the paper (user, 2026-09-29)
  F  discount design and rate paths        bbl_discount.env (utils/bbl_discount.py): beta, T and the
                                           mean rate beta comes from; cf_deposit_sim.jl
                                           RDEP_STABILITY_KAPPA: the deposit-rate cap;
                                           COST_FWD/forward_rf_vintages.csv: each Focus vintage's
                                           neutral rate and the discount weight of its flat tail
  H  descriptives                         Rout/Compressed_DFirm_PrePost_Pix_*.csv (Table 2, written
                                           with it by make_desc_compressed_tables.py) and
                                           Rout/cluster_imbalance.json (Table 3 and the sleepiness
                                           sample's G*, sleep_desc_clusters.py)
  I  policy-function fit (C.10, C.11)      COST_POLFUNC/polfunc_summary.json (bbl_polfunc.py): R^2,
                                           mean spread, N, G, G* per column
  J  entry events (Figure 2)               DIAG_PHI_SEPARATION/d6_entry_paths.csv, n at h = 0
  G  table rows the prose quotes           Table 4: Rout/est1-4_spec12_stage2_comparison.json, the
                                           unrounded sidecar sleep_export_spec12_compare.py writes
                                           with it; Tables 5-6 as rendered, at their three decimals
                                           (their generators export no unrounded values for these
                                           rows): demand coefficients and standard errors, stars,
                                           sample rows; the Pix effect relative to the awake share

CROSS-CHECK. Every number that a rendered table also prints (Drafts/Deposit Competition) is
compared with the printed cell after rounding to three decimals; a cell that differs or cannot be
found is a MISMATCH, listed in both outputs, and makes the script exit with status 1.

STATUS of an entry: ok; pending-_ms982 (a BBL number of a run other than the target multi-start
run); pending-_sc982 (T11's single-curve column from a run other than the target single-curve
run); pending-phi-summary (the phi path summary is not written yet); dropped.

MACROS. Every run also writes paper_numbers_macros.tex, to PAPER_NUMBERS and to Drafts next to the
tables, so the paper quotes a number as \pn{<id>[.<leaf>]} (\input the file in the preamble,
after xcolor and hyperref). \pn{key} prints the tables' three decimals (counts as integers);
\pn[d]{key} d = 0-3 decimals; \pn[ad]{key} the absolute value; \pn[pd]{key} 100 x the value (a
share as percent); \pn[w]{key} |value| rounded to an integer, in words; \pn[w10]{key} rounded to
tens, in words. A pending, dropped or missing number, an unknown format, and a coarser rounding of
a 3-decimal table value (G) that depends on the digit the table dropped print a red ?? and a LaTeX
warning; \PNshowpendingtrue shows a pending number's provisional value in red. Negatives carry a
math minus in text and math alike; \pn is robust in captions and falls back to the 3-decimal
value in PDF bookmarks.

After the _ms982 (and _sc982) solves are ingested, one line each, from the repo folder:
  python make_bbl_cost_tables.py --psi-tag _ms982 --compare-single-curve --single-curve-tag _sc982
  python make_bbl_cost_tables.py --psi-tag _ms982 --from-psi --psi-zip <the run's bbl_outputs zip>
  python make_paper_numbers.py

Usage:
  python make_paper_numbers.py                     # the target run's numbers when present, else
                                                   # the newest bbl_table_numbers_ms*.json
  python make_paper_numbers.py --bbl-tag _ms1      # pin the BBL run
  python make_paper_numbers.py --no-tex            # skip paper_numbers_macros.tex
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd

import make_blp_rc_table as rc          # load_stage, alpha_of, mean_rho_one_minus_s
from utils import paths as _paths
from utils import routines as _routines
from utils import state_transform as _st
from utils.window import MIN_YEAR, MAX_YEAR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EST_OUT = _paths.estimation_output()
COST_FWD = _paths.cost_fwd_dir()
DRAFTS = _paths.drafts_dir()
TABLES_DIR = DRAFTS                     # where the cross-check reads the rendered tables
OUT_DIR = EST_OUT / "PAPER_NUMBERS"
LOGIT_SUMMARY = _paths.blp_results_dir() / "logit" / "logit_summary_spec_12.json"
BLP_STAGE = "ext1"                      # Table 6 (make_blp_demand_comparison_table STAGE_SPECS)
LOGIT_SUBMODEL = "full"                 # Table 5 (blp_logit.jl COMPARISON_SUBMODEL)
D6_DIR = EST_OUT / "DIAG_PHI_SEPARATION"
ROUTINES = tuple(_routines.LINK_ESTS)   # E3, E4
LINEAR_ROUTINES = tuple(e for e in _routines.ACTIVE if e not in _routines.LINK_ESTS)   # E1, E2
BLOCKS = ("B", "D")
PHI_HORIZONS = {0: "t=0", 40: "X (t=40)", 250: "Y (t=250)"}

# The rendered tables the cross-check reads (V_Main's \input names).
T4 = "est1-4_spec12_stage2_comparison.tex"
T5 = "est1-4_spec12_logit_comparison.tex"
T6 = "blp_demand_comparison_noseg_spec12.tex"
T7 = "tab_bbl_cost_identified.tex"
T8 = "tab_bbl_ridge_diagnostic.tex"
T9 = "tab_bbl_cbar.tex"
T10 = "tab_bbl_ridge_by_start.tex"
T11 = "tab_bbl_cbar_design.tex"
TUD = "tab_bbl_violated_by_sign.tex"
T2 = "Compressed_DFirm_PrePost_Pix.tex"
T3 = "cluster_imbalance_panelB.tex"
TC10 = "polfunc_k4.tex"
TC11 = "polfunc_k5.tex"


# ── display ──────────────────────────────────────────────────────────────────
def fmt3(x) -> str:
    """Three decimals, never -0.000 (the tables' display rule). No thousands separator: this is
    the form the cross-check compares."""
    s = f"{float(x):.3f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def fmt3s(x) -> str:
    """fmt3 with thousands separators, for display."""
    s = f"{float(x):,.3f}"
    return s[1:] if s.startswith("-") and float(s.replace(",", "")) == 0.0 else s


def fmt_count(n) -> str:
    return f"{int(n):,}"


def round3(x):
    """A value as the entry's `value`: floats at three decimals (never -0.0), everything else as
    it is; the unrounded value is kept in `value_full`."""
    if isinstance(x, dict):
        return {k: round3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round3(v) for v in x]
    if isinstance(x, bool) or not isinstance(x, float):
        return x
    return float(fmt3(x)) if math.isfinite(x) else None


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


# ── routine labels as the paper prints them ──────────────────────────────────
def roman_of() -> dict:
    r"""{routine: '(III)'}: the position of each \item\label{estimation:*} in V_Main's
    enumerate(label=(\Roman*)), matched to est_ref. Falls back to the routine number."""
    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    out = {E: f"({romans[E - 1]})" for E in _routines.ACTIVE if 0 < E <= len(romans)}
    vm = DRAFTS / "V_Main.tex"
    try:
        labels = re.findall(r"\\item\s*\\label\{estimation:([A-Za-z_]+)\}",
                            vm.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return out
    for E in _routines.ACTIVE:
        m = re.search(r"\\ref\{estimation:([A-Za-z_]+)\}", _routines.est_ref(E))
        if m and m.group(1) in labels:
            out[E] = f"({romans[labels.index(m.group(1))]})"
    return out


ROMAN = roman_of()


def rname(E) -> str:
    return f"E{E} {ROMAN.get(E, '')}".strip()


# ── rendered-table parsing (the cross-check; a source only for the rows of G) ─
_SCI = re.compile(r"(-?\d+\.\d+)\s*\\times\s*10\^\{(-?\d+)\}")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_REF = re.compile(r"\\ref\{estimation:([A-Za-z_]+)\}")
_FIRM = re.compile(r"\((B|D)\)")


def cell_numbers(cell: str) -> list:
    """The numbers one table cell prints, in order: math-mode thousands separators ({,}) dropped,
    a \\times 10^{e} mantissa expanded, macros such as \\ddagger ignored."""
    s = cell.replace("{,}", "")
    out = [float(m.group(1)) * 10.0 ** int(m.group(2)) for m in _SCI.finditer(s)]
    s = re.sub(r"\\[A-Za-z]+", " ", _SCI.sub(" ", s))
    out += [float(m.group(0).replace(",", "")) for m in _NUM.finditer(s)]
    return out


_MULTIROW = re.compile(r"^\\multirow\[[a-z]\]\{\d+\}\{[^}]*\}\{\\raggedright\s*")


def row_label(lab: str) -> str:
    """A body row's label as printed: the \\multirow[t]{2}{<width>}{\\raggedright ...} wrapper of
    the two-line coefficient rows removed."""
    s = lab.strip()
    m = _MULTIROW.match(s)
    return s[m.end():-1].strip() if m and s.endswith("}") else s


def _ref_to_E() -> dict:
    return {_REF.search(_routines.est_ref(E)).group(1): E for E in _routines.ACTIVE
            if _REF.search(_routines.est_ref(E))}


REF_E = _ref_to_E()


class Rendered:
    """One rendered table: its column keys and its body rows (panel, label, cells)."""

    def __init__(self, name: str):
        self.name, self.path = name, TABLES_DIR / name
        self.ok = self.path.is_file()
        self.cols, self.rows = [], []
        if self.ok:
            self._parse(self.path.read_text(encoding="utf-8", errors="replace").splitlines())

    @staticmethod
    def _split(line: str) -> list:
        line = re.sub(r"\\\\\*?\s*(\\cmidrule.*)?$", "", line.strip())
        return [c.strip() for c in re.split(r"(?<!\\)&", line)]

    @classmethod
    def _firm_spans(cls, line: str) -> list:
        """The firm type (B/D) of each column under a header line whose every cell names one,
        e.g. ' & \\multicolumn{2}{c}{Brick-and-Mortar (B)} & \\multicolumn{2}{c}{Digital (D)}';
        [] for any other line."""
        out = []
        for c in cls._split(line)[1:]:
            span, firm = re.match(r"\\multicolumn\{(\d+)\}", c), _FIRM.search(c)
            if not firm:
                return []
            out += [firm.group(1)] * (int(span.group(1)) if span else 1)
        return out

    def _parse(self, lines):
        # Header: the first line naming the routines, expanded over \multicolumn spans, and the
        # line under it when it splits each routine into sub-columns (B/D, single/multi), or the
        # line above it when it groups the routines by firm type (Table 7: B | D side by side).
        hi = next((i for i, l in enumerate(lines) if _REF.search(l)), None)
        if hi is not None:
            cols = []
            for c in self._split(lines[hi])[1:]:
                m, span = _REF.search(c), re.match(r"\\multicolumn\{(\d+)\}", c)
                cols += [REF_E.get(m.group(1)) if m else None] * (int(span.group(1)) if span else 1)
            sub = self._split(lines[hi + 1])[1:] if hi + 1 < len(lines) else []
            over = self._firm_spans(lines[hi - 1]) if hi > 0 else []
            if len(sub) == len(cols) and all(s and not _REF.search(s) for s in sub):
                cols = [(E, s) for E, s in zip(cols, sub)]
            elif over and len(over) == len(cols):
                cols = [(E, k) for E, k in zip(cols, over)]
            self.cols = cols
        start = next((i for i, l in enumerate(lines) if r"\endlastfoot" in l), None)
        if start is None:
            start = next((i for i, l in enumerate(lines) if r"\endhead" in l), 0)
        panel = ""
        for raw in lines[start + 1:]:
            line = raw.strip()
            if line.startswith(r"\end{xltabular}"):
                break
            if line.startswith(r"\multicolumn") and "Panel" in line:
                panel = line
                continue
            if "&" not in line or line.startswith(r"\multicolumn") or not re.search(r"\\\\\*?\s*$", line):
                continue
            cells = self._split(line)
            self.rows.append((panel, cells[0], cells[1:]))

    def find(self, label, panel=None, after=None, nth=0, exact=False):
        """Cells of the nth row whose label starts with `label` (equals it when `exact`), inside
        the panel whose header contains `panel`, and below the first row labelled `after`."""
        seen_after = after is None
        k = 0
        for p, lab, cells in self.rows:
            if panel is not None and panel not in p:
                continue
            if not seen_after:
                seen_after = lab.startswith(after)
                continue
            if (lab == label) if exact else lab.startswith(label):
                if k == nth:
                    return cells
                k += 1
        return None

    def find_row(self, label, exact=False):
        """(cells, cells of the unlabelled row under it) of the first row whose label, with any
        \\multirow wrapper removed, starts with `label` (equals it when `exact`); (None, None)
        when there is none."""
        for i, (_, lab, cells) in enumerate(self.rows):
            if (row_label(lab) == label) if exact else row_label(lab).startswith(label):
                nxt = self.rows[i + 1] if i + 1 < len(self.rows) else None
                return cells, (nxt[2] if nxt is not None and not nxt[1].strip() else None)
        return None, None

    def cell(self, cells, key):
        if cells is None or key not in self.cols:
            return None
        i = self.cols.index(key)
        return cells[i] if i < len(cells) else None


_TABLES: dict = {}


def table(name: str) -> Rendered:
    if name not in _TABLES:
        _TABLES[name] = Rendered(name)
    return _TABLES[name]


def check_cell(tname, where, text, expected) -> dict:
    """One cross-check: the numbers `expected` (display values) against the printed cell."""
    if any(not _finite(v) for v in expected):
        return dict(table=tname, cell=where, expected=None, printed=None, raw=text, ok=False,
                    why="no value to compare")
    exp = [fmt3(v) for v in expected]
    got = None if text is None else [fmt3(v) for v in cell_numbers(text)][:len(exp)]
    ok = got is not None and got == exp
    return dict(table=tname, cell=where, expected=exp, printed=got, raw=text,
                ok=ok, why=None if ok else ("cell not found" if text is None else "differs"))


# ── entries ──────────────────────────────────────────────────────────────────
ENTRIES: list = []


def add(id_, plan, label, value, definition, source, status, run=None, display=None,
        detail=None, checks=None, unit=None, rounded3=False):
    """One entry. `rounded3`: the value was read at three decimals (a rendered table cell), so the
    macros refuse a coarser rounding that depends on a digit the table rounded away."""
    ENTRIES.append(dict(id=id_, plan=plan, label=label, value=round3(value), value_full=value,
                        display=display, unit=unit, definition=definition, source=source, run=run,
                        status=status, detail=detail, checks=checks or [], rounded3=rounded3))


def disp(v, kind="num") -> str:
    if v is None:
        return "--"
    if kind == "count":
        return fmt_count(v)
    if kind == "share":
        return f"{fmt3s(v)} ({fmt3s(100.0 * float(v))}%)"
    return fmt3s(v)


def disp_ci(lo, hi) -> str:
    return f"[{fmt3s(lo)}, {fmt3s(hi)}]"


# ── A: BBL ───────────────────────────────────────────────────────────────────
def bbl_file(tag: str | None, target: str):
    """The bbl_table_numbers file to read: `tag` when given, else the target run's, else the most
    recently written multi-start one. -> (path | None, why)."""
    if tag is not None:
        p = COST_FWD / f"bbl_table_numbers{tag}.json"
        return (p if p.is_file() else None), f"pinned --bbl-tag {tag}"
    p = COST_FWD / f"bbl_table_numbers{target}.json"
    if p.is_file():
        return p, f"the target run {target}"
    cands = sorted(COST_FWD.glob("bbl_table_numbers_ms*.json"), key=lambda q: q.stat().st_mtime)
    if cands:
        return cands[-1], f"target {target} absent; newest multi-start file"
    return None, "no bbl_table_numbers_ms*.json in COST_FWD"


def _cost_params_theta(files: dict) -> dict:
    """{'E3_B': [omega, zeta]} of the cost_params files a section names, read from the folder the
    ingest writes them to, for the staleness check."""
    out = {}
    d = _paths.bbl_output_dir() / "cluster_processed"
    for Ek, name in (files or {}).items():
        if not name or not (d / name).is_file():
            continue
        cp = json.loads((d / name).read_text(encoding="utf-8"))
        for k in BLOCKS:
            if cp.get(k):
                out[f"{Ek}_{k}"] = [float(cp[k]["omega"]), float(cp[k]["zeta"])]
    return out


def collect_bbl(doc: dict, path: pathlib.Path, target: str, sc_target: str, notes: list):
    secs = doc.get("sections") or {}
    tag = doc.get("psi_tag")
    status = "ok" if tag == target else f"pending-{target}"
    src = path.name

    # Every section must describe the same solve, and that solve must be the cost_params on disk.
    thetas = {n: s.get("theta") for n, s in secs.items() if s.get("theta") is not None}
    ref = (secs.get("cost_tables") or {}).get("theta")
    for n, th in thetas.items():
        if ref is not None and th != ref:
            notes.append(f"STALE: section {n} of {src} describes another solve than cost_tables "
                         f"(theta differs); re-run the pass that writes it.")
    on_disk = _cost_params_theta((secs.get("cost_tables") or {}).get("files"))
    if ref is not None and on_disk and any(on_disk.get(k) != v for k, v in ref.items()):
        notes.append(f"STALE: {src} was written from a cost_params that is no longer the one in "
                     f"BBL_OUTPUT/cluster_processed; re-run both passes of make_bbl_cost_tables.py.")
    for need in ("cost_tables", "tab_bbl_ridge_diagnostic", "tab_bbl_ridge_by_start",
                 "tab_bbl_violated_by_sign", "tab_bbl_cbar_design"):
        if need not in secs:
            notes.append(f"{src} has no section {need}: run the make_bbl_cost_tables.py pass that "
                         f"writes it (see the module docstring).")

    def prov(sec, E):
        p = ((secs.get(sec) or {}).get("provenance") or {}).get(f"E{E}") or {}
        return dict(tag=p.get("tag"), beta=p.get("beta"), T=p.get("T"), source=p.get("source"))

    def run_of(sec, Es):
        return [dict(routine=rname(E), **prov(sec, E)) for E in Es]

    # ── Table 7 ──
    ct = secs.get("cost_tables") or {}
    blocks = ct.get("blocks") or {}
    t7 = table(T7)
    thr = next((b.get("cond_max") for b in blocks.values() if b.get("cond_max")), None)
    if thr is not None:
        add("bbl.t7.cond_gate", "3.5(a) Table 7", "Condition-index gate of Table 7", thr,
            "ridge_cond_max: the pooled condition index above which a block's omega and zeta are "
            "not read separately (bbl_solve.py --ridge-cond-max). A setting of the solve, the "
            "same for every run", f"{src} sections.cost_tables.blocks.*.cond_max", "ok",
            display=disp(thr))
    panel = {"B": "(B) Firms", "D": "(D) Firms"}
    # Table 7 prints B and D side by side (columns keyed (routine, firm type), row labels shared)
    # or stacked in panels (columns keyed by routine); both are read.
    side = any(isinstance(c, tuple) for c in t7.cols)

    def t7_row(label, k, after=None):
        return t7.find(label, after=after) if side else t7.find(label, panel=panel[k], after=after)

    def t7_cell(cells, E, k):
        return t7.cell(cells, (E, k) if side else E)

    dag_cells = []
    for E in ROUTINES:
        for k in BLOCKS:
            b = blocks.get(f"E{E}_{k}")
            if not b:
                continue
            c = b.get("cond_pooled")
            flagged = b.get("identified") is False
            if flagged:
                dag_cells.append((E, k, c))
            cells = t7_row("Condition index", k)
            add(f"bbl.t7.cond.E{E}.{k}", "3.5(a) Table 7",
                f"Condition index, {k} firms, {rname(E)}", c,
                "Pooled Belsley-Kuh-Welsch condition index of [dpsi_2 dpsi_4] in the block (columns "
                "scaled to unit length); a block is marked double-dagger when the solve's "
                "identified_split is false, i.e. the index exceeds the threshold "
                f"(ridge_cond_max = {fmt3(thr) if thr else '--'}).",
                f"{src} sections.cost_tables.blocks.E{E}_{k}.cond_pooled", status,
                run=run_of("cost_tables", [E]), display=disp(c) + (" ‡" if flagged else ""),
                detail=dict(threshold=thr, double_dagger=flagged, identified_split=b.get("identified")),
                checks=[check_cell(T7, f"{k} firms, Condition index, {rname(E)}",
                                   t7_cell(cells, E, k), [c])])
    # Which cells carry the double dagger, as printed.
    printed = set()
    for k in BLOCKS:
        cells = t7_row("Condition index", k)
        for E in ROUTINES:
            txt = t7_cell(cells, E, k)
            if txt and "ddagger" in txt:
                printed.add((E, k))
    want = {(E, k) for E, k, _ in dag_cells}
    bad = [f"{rname(E)} {k}: condition index {fmt3(c)} above {fmt3(thr)}" for E, k, c in dag_cells]
    incons = [f"{rname(E)} {k}" for E, k, c in dag_cells if thr and c is not None and c <= thr]
    add("bbl.t7.dagger", "3.5(a) Table 7", "Cells marked ‡ in Table 7 and why",
        [f"{rname(E)} {k}" for E, k, _ in dag_cells],
        "The blocks whose solve records identified_split = false, which Table 7 marks with a "
        "double dagger (omega and zeta not separately identified there; only c-bar is "
        "interpretable). The reason is the pooled condition index above the threshold.",
        f"{src} sections.cost_tables.blocks.*.identified / cond_pooled / cond_max", status,
        run=run_of("cost_tables", ROUTINES),
        display="; ".join(bad) if bad else "none",
        detail=dict(n_cells=len(dag_cells), threshold=thr,
                    verdict_contradicts_index=incons or None),
        checks=[dict(table=T7, cell="double-dagger cells of the Condition index row",
                     expected=sorted(f"E{E}{k}" for E, k in want),
                     printed=sorted(f"E{E}{k}" for E, k in printed), raw=None,
                     ok=printed == want, why=None if printed == want else "differs")])
    present = [(E, k) for E in ROUTINES for k in BLOCKS if blocks.get(f"E{E}_{k}")]
    if present:
        n_id = len(present) - len(dag_cells)
        add("bbl.t7.blocks", "3.5(a) Table 7", "Blocks that clear the collinearity gate",
            dict(identified=n_id, total=len(present), failed=len(dag_cells)),
            "Number of routine x firm-type blocks whose solve records identified_split (omega and "
            "zeta separately identified; the 'three of the four blocks' of the prose), of the "
            "blocks solved, and the number marked double-dagger",
            f"{src} sections.cost_tables.blocks.*.identified", status,
            run=run_of("cost_tables", ROUTINES),
            display=f"{n_id} of {len(present)} identified")

    def est_ci(E, k, par):
        b = blocks.get(f"E{E}_{k}") or {}
        ci = b.get(f"ci_{par}") or {}
        return b.get(par), ci.get("ci_lo"), ci.get("ci_hi"), ci

    # Row-label prefixes: $\hat\omega^{\mathrm{B}}$ (stacked) and $\hat\omega^{\kappa}$ (side by side).
    sym = {"omega": r"$\hat\omega", "zeta": r"$\hat\zeta"}
    for par, word in (("zeta", "zeta"), ("omega", "omega")):
        for E in ROUTINES:
            v, lo, hi, ci = est_ci(E, "B", par)
            if v is None:
                continue
            lab = sym[par]
            est_cells = t7_row(lab, "B")
            ci_cells = t7_row(r"\quad 95\% CI", "B", after=lab)
            inc0 = (lo <= 0.0 <= hi) if _finite(lo) and _finite(hi) else None
            add(f"bbl.t7.{par}.E{E}.B", "3.5(a) Table 7", f"{word}^B, {rname(E)}",
                dict(estimate=v, ci=[lo, hi], ci_includes_zero=inc0),
                f"{word}-hat for B firms (native per-quarter units) and its 95% interval from "
                "inverting the eq:16 criterion at a subsampled critical value"
                + (" (ci_includes_zero: whether 0 lies in the interval)" if par == "omega" else ""),
                f"{src} sections.cost_tables.blocks.E{E}_B.{par}, .ci_{par}.ci_lo/ci_hi", status,
                run=run_of("cost_tables", [E]),
                display=f"{disp(v)} {disp_ci(lo, hi) if _finite(lo) else '--'}"
                        + (f"; includes 0: {'yes' if inc0 else 'no'}" if par == "omega" else ""),
                detail=dict(truncated=bool(ci.get("truncated_lo") or ci.get("truncated_hi")),
                            empty=ci.get("empty")),
                checks=[check_cell(T7, f"B firms, {word}, {rname(E)}",
                                   t7_cell(est_cells, E, "B"), [v]),
                        check_cell(T7, f"B firms, {word} 95% CI, {rname(E)}",
                                   t7_cell(ci_cells, E, "B"), [lo, hi])])
        if par == "zeta":
            los = [est_ci(E, "B", "zeta")[1] for E in ROUTINES]
            his = [est_ci(E, "B", "zeta")[2] for E in ROUTINES]
            if all(_finite(x) for x in los + his):
                add("bbl.t7.zeta.B.ci_span", "3.5(a) Table 7",
                    "zeta^B intervals, lowest lower and highest upper bound across routines",
                    [min(los), max(his)],
                    "min over routines of the zeta^B interval's lower bound and max of its upper "
                    "bound (the 'about 0.3-2.2' of the prose)",
                    f"{src} sections.cost_tables.blocks.E*_B.ci_zeta", status,
                    run=run_of("cost_tables", ROUTINES), display=disp_ci(min(los), max(his)))
    E0 = ROUTINES[0]
    v, lo, hi, _ = est_ci(E0, "D", "zeta")
    if v is not None:
        add(f"bbl.t7.zeta.E{E0}.D.ci", "3.5(a) Table 7", f"zeta^D interval, {rname(E0)}", [lo, hi],
            "95% interval of zeta-hat for D firms from inverting the eq:16 criterion",
            f"{src} sections.cost_tables.blocks.E{E0}_D.ci_zeta.ci_lo/ci_hi", status,
            run=run_of("cost_tables", [E0]), display=disp_ci(lo, hi), detail=dict(estimate=v),
            checks=[check_cell(T7, f"D firms, zeta 95% CI, {rname(E0)}",
                               t7_cell(t7_row(r"\quad 95\% CI", "D", after=sym["zeta"]), E0, "D"),
                               [lo, hi])])
    # D firms: the slope and the intercept, per routine (the interval of the first routine's slope
    # is the entry above).
    for E in ROUTINES:
        v, lo, hi, ci = est_ci(E, "D", "zeta")
        if v is not None:
            add(f"bbl.t7.zeta.E{E}.D.estimate", "3.5(a) Table 7", f"zeta^D, {rname(E)}", v,
                "zeta-hat for D firms (native per-quarter units)",
                f"{src} sections.cost_tables.blocks.E{E}_D.zeta", status,
                run=run_of("cost_tables", [E]), display=disp(v),
                checks=[check_cell(T7, f"D firms, zeta, {rname(E)}",
                                   t7_cell(t7_row(sym["zeta"], "D"), E, "D"), [v])])
            if E != E0:
                add(f"bbl.t7.zeta.E{E}.D.ci", "3.5(a) Table 7", f"zeta^D interval, {rname(E)}",
                    [lo, hi], "95% interval of zeta-hat for D firms from inverting the eq:16 criterion",
                    f"{src} sections.cost_tables.blocks.E{E}_D.ci_zeta.ci_lo/ci_hi", status,
                    run=run_of("cost_tables", [E]), display=disp_ci(lo, hi),
                    checks=[check_cell(T7, f"D firms, zeta 95% CI, {rname(E)}",
                                       t7_cell(t7_row(r"\quad 95\% CI", "D", after=sym["zeta"]), E, "D"),
                                       [lo, hi])])
        v, lo, hi, ci = est_ci(E, "D", "omega")
        if v is not None:
            inc0 = (lo <= 0.0 <= hi) if _finite(lo) and _finite(hi) else None
            add(f"bbl.t7.omega.E{E}.D", "3.5(a) Table 7", f"omega^D, {rname(E)}",
                dict(estimate=v, ci=[lo, hi], ci_includes_zero=inc0),
                "omega-hat for D firms (native per-quarter units) and its 95% interval from inverting "
                "the eq:16 criterion (ci_includes_zero: whether 0 lies in the interval)",
                f"{src} sections.cost_tables.blocks.E{E}_D.omega, .ci_omega.ci_lo/ci_hi", status,
                run=run_of("cost_tables", [E]),
                display=f"{disp(v)} {disp_ci(lo, hi) if _finite(lo) else '--'}"
                        f"; includes 0: {'yes' if inc0 else 'no'}",
                checks=[check_cell(T7, f"D firms, omega, {rname(E)}",
                                   t7_cell(t7_row(sym["omega"], "D"), E, "D"), [v]),
                        check_cell(T7, f"D firms, omega 95% CI, {rname(E)}",
                                   t7_cell(t7_row(r"\quad 95\% CI", "D", after=sym["omega"]), E, "D"),
                                   [lo, hi])])

    def cmed_entry(id_, plan, what, b, tname, tab, pan, k, E, src_field, st, run):
        """The cost at the median shifters of one block, c-bar + gamma'Z-bar, read from the block's
        cbar_at_median_shifters record. -> the record when it holds an estimate, else None."""
        cm = (b or {}).get("cbar_at_median_shifters") or {}
        mv, msd = cm.get("estimate_ann"), cm.get("sd_ann")
        mlo, mhi = cm.get("ci_lo_ann"), cm.get("ci_hi_ann")
        if cm.get("status") != "ok" or not _finite(mv):
            return None
        mexcl = (mlo > 0.0 or mhi < 0.0) if _finite(mlo) and _finite(mhi) else None
        lab = r"$\hat{\bar c}^{\mathrm{%s}}+" % k
        add(id_, plan, f"{what}, {k} firms, {rname(E)}",
            dict(estimate=mv, sd=msd, ci=[mlo, mhi], ci_excludes_zero=mexcl),
            "c-bar + gamma'Z-bar = omega + rbar_f * zeta + gamma'Z-bar, the marginal cost of deposits "
            "of a firm with median cost shifters: Z-bar is the median, shifter by shifter, of "
            "dpsi_3/dpsi_2 over the rows rbar_f is taken over. Compounded annual pp; sd: firm-block "
            "bootstrap (delta method); ci: subsampling 95% interval; ci_excludes_zero: whether the "
            "interval excludes 0",
            f"{src} {src_field}.cbar_at_median_shifters.estimate_ann, .sd_ann, .ci_lo_ann/.ci_hi_ann",
            st, run=run, unit="compounded annual pp",
            display=f"{disp(mv)} ({disp(msd)}) {disp_ci(mlo, mhi) if _finite(mlo) else '--'}; "
                    f"excludes 0: {'yes' if mexcl else 'no'}",
            detail=dict(zbar=cm.get("zbar"), gamma_zbar=cm.get("gamma_zbar"), rbar=cm.get("rbar"),
                        psi_source=cm.get("psi_source")),
            checks=[check_cell(tname, f"Panel {k}, cost at median shifters, {rname(E)}",
                               tab.cell(tab.find(lab, panel=pan), E), [mv]),
                    check_cell(tname, f"Panel {k}, cost at median shifters SD, {rname(E)}",
                               tab.cell(tab.find(r"\quad {\scriptsize at median", panel=pan), E),
                               [msd]),
                    check_cell(tname, f"Panel {k}, cost at median shifters 95% CI, {rname(E)}",
                               tab.cell(tab.find(r"\quad 95\% CI", panel=pan, after=lab), E),
                               [mlo, mhi])])
        return cm

    def cmed_count(id_, plan, what, recs, src_field, st, run=None):
        """Signs and intervals of the cost at the median shifters, counted over the blocks."""
        if not recs:
            return

        def _x(c, side):
            lo, hi = c.get("ci_lo_ann"), c.get("ci_hi_ann")
            return _finite(lo) and _finite(hi) and (lo > 0.0 if side > 0 else hi < 0.0)
        cnt = dict(total=len(recs), positive=sum(c["estimate_ann"] > 0.0 for c in recs.values()))
        for k in BLOCKS:
            of_k = [c for (E, k2), c in recs.items() if k2 == k]
            cnt[f"{k}_blocks"] = len(of_k)
            cnt[f"{k}_positive"] = sum(c["estimate_ann"] > 0.0 for c in of_k)
            cnt[f"{k}_ci_above_zero"] = sum(_x(c, +1) for c in of_k)
            cnt[f"{k}_ci_below_zero"] = sum(_x(c, -1) for c in of_k)
        add(id_, plan, f"{what}: signs and intervals, counted", cnt,
            "Over the routine x firm-type blocks: how many estimates of the cost at the median "
            "shifters are positive (in all and per firm type), and per firm type how many 95% "
            "intervals lie entirely above zero and entirely below zero",
            f"{src} {src_field}.*.cbar_at_median_shifters", st, run=run,
            display=f"{cnt['positive']} of {cnt['total']} positive; "
                    + "; ".join(f"{k}: {cnt[f'{k}_ci_above_zero']} above 0, "
                                f"{cnt[f'{k}_ci_below_zero']} below 0, of {cnt[f'{k}_blocks']}"
                                for k in BLOCKS))

    # ── Table 9 ──
    t9 = table(T9)
    rng = {}
    cmeds = {}
    for E in ROUTINES:
        for k in BLOCKS:
            b = blocks.get(f"E{E}_{k}")
            if not b:
                continue
            p = panel[k]
            fb = b.get("frac_bind")
            add(f"bbl.t9.violated.E{E}.{k}", "3.5(c) Table 9",
                f"Share of inequalities violated, {k} firms, {rname(E)}", fb,
                "Share of the eq:16 deviation inequalities with g < 0 at theta-hat (last row of "
                "Table 9; 1/2 is the mechanical value)",
                f"{src} sections.cost_tables.blocks.E{E}_{k}.frac_bind", status,
                run=run_of("cost_tables", [E]), display=disp(fb, "share"),
                checks=[check_cell(T9, f"Panel {k}, violated share, {rname(E)}",
                                   t9.cell(t9.find("Share of deviation", panel=p), E), [fb])])
            rb = b.get("rbar_ann")
            add(f"bbl.t9.rbar.E{E}.{k}", "3.5(d) Table 9", f"rbar_f^{k}, {rname(E)}", rb,
                "Median over the type's firm x launch-quarter x deviation rows (dpsi_2 != 0, dead "
                "firm-quarters out) of dpsi_4/dpsi_2, the forward rate a deviation's deposits are "
                "priced at; compounded annual pp, ((1+x)^4-1)*100",
                f"{src} sections.cost_tables.blocks.E{E}_{k}.rbar_ann (quarterly: .rbar)", status,
                run=run_of("cost_tables", [E]), display=disp(rb), unit="compounded annual pp",
                checks=[check_cell(T9, f"Panel {k}, rbar_f, {rname(E)}",
                                   t9.cell(t9.find(r"$\bar r^{f,", panel=p), E), [rb])])
            cb, lo, hi = b.get("cbar_ann"), b.get("ci_lo_ann"), b.get("ci_hi_ann")
            excl = (lo > 0.0 or hi < 0.0) if _finite(lo) and _finite(hi) else None
            add(f"bbl.t9.cbar.E{E}.{k}", "3.5(d) Table 9", f"c-bar^{k}, {rname(E)}",
                dict(estimate=cb, ci=[lo, hi], ci_excludes_zero=excl),
                "c-bar = omega + rbar_f * zeta, the marginal cost of deposits at the type's forward "
                "rate, and its subsampling 95% interval; compounded annual pp. ci_excludes_zero: "
                "whether the interval excludes 0",
                f"{src} sections.cost_tables.blocks.E{E}_{k}.cbar_ann, .ci_lo_ann/.ci_hi_ann",
                status, run=run_of("cost_tables", [E]), unit="compounded annual pp",
                display=f"{disp(cb)} {disp_ci(lo, hi) if _finite(lo) else '--'}; "
                        f"excludes 0: {'yes' if excl else 'no'}",
                detail=dict(sd_ann=b.get("cbar_sd_ann"), ci_rbar_ann=b.get("ci_rbar_ann")),
                checks=[check_cell(T9, f"Panel {k}, c-bar, {rname(E)}",
                                   t9.cell(t9.find(r"$\hat{\bar c}", panel=p), E), [cb]),
                        check_cell(T9, f"Panel {k}, c-bar 95% CI, {rname(E)}",
                                   t9.cell(t9.find(r"\quad 95\% CI", panel=p), E), [lo, hi])])
            cm = cmed_entry(f"bbl.t9.cmed.E{E}.{k}", "3.5(d) Table 9", "Cost at median shifters", b,
                            T9, t9, p, k, E, f"sections.cost_tables.blocks.E{E}_{k}", status,
                            run_of("cost_tables", [E]))
            if cm:
                cmeds[(E, k)] = cm
            for key, val in (("violated", fb), ("rbar", rb), ("cbar", cb),
                             ("cmed", cm.get("estimate_ann") if cm else None)):
                rng.setdefault((key, k), []).append(val)
    fbs = [v for k in BLOCKS for v in rng.get(("violated", k), []) if _finite(v)]
    if fbs:
        add("bbl.t9.violated.range", "3.5(c) Table 9", "Violated shares, range over blocks",
            [min(fbs), max(fbs)], "min and max of the violated share over routine x firm type",
            f"{src} sections.cost_tables.blocks.*.frac_bind", status,
            run=run_of("cost_tables", ROUTINES), display=disp_ci(min(fbs), max(fbs)))
    for key, what in (("rbar", "rbar_f"), ("cbar", "c-bar"),
                      ("cmed", "cost at median shifters")):
        for k in BLOCKS:
            vals = [v for v in rng.get((key, k), []) if _finite(v)]
            if vals:
                add(f"bbl.t9.{key}.{k}.range", "3.5(d) Table 9",
                    f"{what}^{k}, range over routines", [min(vals), max(vals)],
                    f"min and max over routines of {what} for {k} firms (compounded annual pp)",
                    f"{src} sections.cost_tables.blocks.E*_{k}."
                    + ("cbar_at_median_shifters.estimate_ann" if key == "cmed" else f"{key}_ann"),
                    status,
                    run=run_of("cost_tables", ROUTINES), display=disp_ci(min(vals), max(vals)),
                    unit="compounded annual pp")
                if key == "rbar":
                    add(f"bbl.t9.rbar.{k}.mean", "3.5(d) Table 9",
                        f"rbar_f^{k}, mean over routines", float(np.mean(vals)),
                        f"mean over routines of rbar_f for {k} firms (the 'about 8.0 percent a "
                        "year' of the prose; compounded annual pp)",
                        f"{src} sections.cost_tables.blocks.E*_{k}.rbar_ann", status,
                        run=run_of("cost_tables", ROUTINES), display=disp(float(np.mean(vals))),
                        unit="compounded annual pp")
    cbs = {(E, k): blocks[f"E{E}_{k}"] for E in ROUTINES for k in BLOCKS
           if blocks.get(f"E{E}_{k}") and _finite(blocks[f"E{E}_{k}"].get("cbar_ann"))}
    if cbs:
        def _excl(b):
            lo, hi = b.get("ci_lo_ann"), b.get("ci_hi_ann")
            return _finite(lo) and _finite(hi) and (lo > 0.0 or hi < 0.0)
        cnt = dict(total=len(cbs), positive=sum(b["cbar_ann"] > 0.0 for b in cbs.values()))
        for k in BLOCKS:
            of_k = [b for (E, k2), b in cbs.items() if k2 == k]
            cnt[f"{k}_blocks"] = len(of_k)
            cnt[f"{k}_excl_zero"] = sum(_excl(b) for b in of_k)
        add("bbl.t9.cbar.count", "3.5(d) Table 9", "c-bar estimates: signs and intervals, counted",
            cnt, "Over the routine x firm-type blocks of Table 9: how many c-bar estimates are "
            "positive (total: blocks with an estimate), and per firm type how many 95% intervals "
            "exclude zero (the 'all four are positive; both D intervals exclude zero and neither "
            "B interval does' of the prose)",
            f"{src} sections.cost_tables.blocks.*.cbar_ann, .ci_lo_ann/.ci_hi_ann", status,
            run=run_of("cost_tables", ROUTINES),
            display=f"{cnt['positive']} of {cnt['total']} positive; "
                    + "; ".join(f"{k}: {cnt[f'{k}_excl_zero']} of {cnt[f'{k}_blocks']} exclude 0"
                                for k in BLOCKS))
    cmed_count("bbl.t9.cmed.count", "3.5(d) Table 9", "Cost at median shifters", cmeds,
               "sections.cost_tables.blocks", status, run_of("cost_tables", ROUTINES))

    # ── appendix up/down ──
    ud = (secs.get("tab_bbl_violated_by_sign") or {}).get("routines") or {}
    tud = table(TUD)
    lab = {"up": "Upward deviations", "down": "Downward deviations"}
    span = {}
    for d in ("up", "down"):
        for E in ROUTINES:
            for k in BLOCKS:
                r = (ud.get(f"E{E}") or {}).get(k)
                if not r:
                    continue
                v = r.get(d)
                span.setdefault(d, []).append(v)
                add(f"bbl.ud.{d}.E{E}.{k}", "3.5(c) appendix up/down",
                    f"Violated share, {d}ward deviations, {k} firms, {rname(E)}", v,
                    f"Share of the {d}ward deviation inequalities ({'raising' if d == 'up' else 'lowering'} "
                    "the firm's spread) failing at theta-hat",
                    f"{src} sections.tab_bbl_violated_by_sign.routines.E{E}.{k}.{d}", status,
                    run=run_of("tab_bbl_violated_by_sign", [E]), display=disp(v, "share"),
                    detail=dict(n=r.get(f"n_{d}")),
                    checks=[check_cell(TUD, f"{lab[d]}, {rname(E)} {k}",
                                       tud.cell(tud.find(lab[d]), (E, k)), [v])])
        vals = [v for v in span.get(d, []) if _finite(v)]
        if vals:
            add(f"bbl.ud.{d}.range", "3.5(c) appendix up/down",
                f"Violated share, {d}ward deviations, range over blocks", [min(vals), max(vals)],
                f"min and max over routine x firm type of the {d}ward violated share",
                f"{src} sections.tab_bbl_violated_by_sign.routines.*.*.{d}", status,
                run=run_of("tab_bbl_violated_by_sign", ROUTINES),
                display=f"{disp_ci(min(vals), max(vals))} "
                        f"({fmt3s(100 * min(vals))}%-{fmt3s(100 * max(vals))}%)")

    # ── Tables 8 and 10 ──
    t8, t10 = table(T8), table(T10)
    rd = (secs.get("tab_bbl_ridge_diagnostic") or {}).get("routines") or {}
    # Cells after the routine label in the trimmed Table 8: n, Corr, Cond. (pooled), Cond. (lowest
    # within a quarter; multi-start psi only), Median, IQR%.
    t8cols = {"corr": 1, "cond": 2, "median": 4}
    for E in ROUTINES:
        s = rd.get(f"E{E}")
        if not s:
            continue
        row = t8.find(_routines.est_ref(E), exact=True)
        pooled10 = t10.find(r"\textit{Pooled}", panel=_routines.est_ref(E) + "}")
        for key, word in (("corr", "correlation"), ("cond", "condition index")):
            v = s.get(key)
            add(f"bbl.t8.{key}.E{E}", "3.4/3.6 Table 8", f"Pooled {word}, {rname(E)}", v,
                f"Pooled {word} of [dpsi_2 dpsi_4] over every row of the routine (firm types and "
                "launch quarters pooled, dead firm-quarters out"
                + ("; Belsley-Kuh-Welsch, columns scaled to unit length)" if key == "cond" else ")"),
                f"{src} sections.tab_bbl_ridge_diagnostic.routines.E{E}.{key}", status,
                run=run_of("tab_bbl_ridge_diagnostic", [E]), display=disp(v),
                checks=[check_cell(T8, f"{rname(E)} row, {word}",
                                   row[t8cols[key]] if row and len(row) > t8cols[key] else None, [v]),
                        check_cell(T10, f"{rname(E)} Pooled row, {word}",
                                   pooled10[{"corr": 2, "cond": 3}[key]]
                                   if pooled10 and len(pooled10) > 3 else None, [v])])
    rbs = (secs.get("tab_bbl_ridge_by_start") or {}).get("routines") or {}
    within = []
    for E in ROUTINES:
        per = {q: s for q, s in (rbs.get(f"E{E}") or {}).items() if q != "pooled"}
        for q, s in per.items():
            if _finite(s.get("cond")):
                within.append((s["cond"], E, q))
        meds = [(s["ratio_median_ann"], q) for q, s in per.items() if _finite(s.get("ratio_median_ann"))]
        if meds:
            for which, (v, q) in (("min", min(meds)), ("max", max(meds))):
                cells = t10.find(q, panel=_routines.est_ref(E) + "}")
                add(f"bbl.t10.median_ratio.E{E}.{which}", "3.4/3.6 Table 10",
                    f"{'Lowest' if which == 'min' else 'Highest'} launch-quarter median "
                    f"dpsi_4/dpsi_2, {rname(E)}", dict(value=v, quarter=q),
                    "Median over the launch quarter's rows (both firm types) of dpsi_4/dpsi_2, "
                    f"compounded annual pp; the {which} over the launch quarters",
                    f"{src} sections.tab_bbl_ridge_by_start.routines.E{E}.<quarter>.ratio_median_ann",
                    status, run=run_of("tab_bbl_ridge_by_start", [E]),
                    display=f"{disp(v)} in {q}", unit="compounded annual pp",
                    checks=[check_cell(T10, f"{rname(E)} {q}, median",
                                       cells[1] if cells and len(cells) > 1 else None, [v])])
    if within:
        for which, (v, E, q) in (("min", min(within)), ("max", max(within))):
            cells = t10.find(q, panel=_routines.est_ref(E) + "}")
            per_r = {rname(r): dict(min=min((c, qq) for c, e_, qq in within if e_ == r),
                                    max=max((c, qq) for c, e_, qq in within if e_ == r))
                     for r in ROUTINES if any(e_ == r for _, e_, _ in within)}
            add(f"bbl.t10.cond_within.{which}", "3.4/3.6 Table 10",
                f"{'Lowest' if which == 'min' else 'Highest'} within-quarter condition index",
                dict(value=v, routine=rname(E), quarter=q),
                "Belsley-Kuh-Welsch condition index of [dpsi_2 dpsi_4] within one launch quarter "
                f"(both firm types); the {which} over every launch quarter of every panel",
                f"{src} sections.tab_bbl_ridge_by_start.routines.E*.<quarter>.cond", status,
                run=run_of("tab_bbl_ridge_by_start", [E]), display=f"{disp(v)} in {q}, {rname(E)}",
                detail={k: {m: [fmt3(c), qq] for m, (c, qq) in d.items()} for k, d in per_r.items()},
                checks=[check_cell(T10, f"{rname(E)} {q}, Cond.",
                                   cells[3] if cells and len(cells) > 3 else None, [v])])

    # ── the runs behind Tables 9 and 11 ──
    def disc_note(tname):
        """beta and T values the table's discount sentence prints."""
        t = table(tname)
        if not t.ok:
            return None
        txt = t.path.read_text(encoding="utf-8", errors="replace")
        return sorted({(fmt3(float(b)), int(n)) for b, n in
                       re.findall(r"\\beta=([0-9.]+)\$.*?\$T=(\d+)\$", txt)})

    def col_entry(id_, label, p, want, tname, sec_name, designs):
        st = "ok" if p.get("tag") == want else f"pending-{want}"
        note = disc_note(tname)
        exp = sorted({(fmt3(q["beta"]), int(q["T"])) for q in designs if q.get("beta") is not None})
        # A table whose Notes print no discount sentence has nothing to cross-check here.
        checks = [] if note == [] else [
            dict(table=tname, cell="discount sentence of the Notes (beta, T)",
                 expected=[list(x) for x in exp],
                 printed=None if note is None else [list(x) for x in note], raw=None,
                 ok=note == exp, why=None if note == exp else "differs")]
        add(id_, "Tables 9/11 columns", label,
            dict(tag=p.get("tag"), beta=p.get("beta"), T=p.get("T")),
            "The psi tag of the run behind the column and the discount factor beta (per quarter) "
            "and horizon T its forward simulation used, from the run's own record",
            f"{src} sections.{sec_name}.provenance ({p.get('source')})", st,
            run=[dict(p)], display=(f"tag {_tag_txt(p.get('tag'))}, beta {fmt3(p['beta'])}, "
                                    f"T {p['T']}" if p.get("beta") is not None else
                                    f"tag {_tag_txt(p.get('tag'))}, "
                                    "no beta/T recorded"),
            checks=checks)

    for E in ROUTINES:
        col_entry(f"bbl.t9.col.E{E}", f"Table 9 column {rname(E)}", prov("cost_tables", E), target,
                  T9, "cost_tables", [prov("cost_tables", E2) for E2 in ROUTINES])
    # Table 11 prints the single-curve columns only (the multi-start ones are Table 9's).
    single = ((secs.get("tab_bbl_cbar_design") or {}).get("single") or {}).get("provenance") or {}
    allp = [single.get(f"E{E2}") or {} for E2 in ROUTINES]
    for E in ROUTINES:
        p = single.get(f"E{E}") or {}
        if p:
            col_entry(f"bbl.t11.col.E{E}.single", f"Table 11 column {rname(E)}, single curve",
                      dict(p), sc_target, T11, "tab_bbl_cbar_design.single", allp)
    sc = (secs.get("tab_bbl_cbar_design") or {}).get("single") or {}
    st11 = "ok" if sc.get("tag") == sc_target else f"pending-{sc_target}"
    t11 = table(T11)
    sblocks = {(E, k): (sc.get("blocks") or {}).get(f"E{E}_{k}") for E in ROUTINES for k in BLOCKS}
    sblocks = {ek: b for ek, b in sblocks.items() if b and _finite(b.get("cbar_ann"))}
    for (E, k), b in sblocks.items():
        cb, lo, hi = b["cbar_ann"], b.get("ci_lo_ann"), b.get("ci_hi_ann")
        excl = (lo > 0.0 or hi < 0.0) if _finite(lo) and _finite(hi) else None
        add(f"bbl.t11.cbar.E{E}.{k}", "Table 11", f"c-bar^{k}, single curve, {rname(E)}",
            dict(estimate=cb, ci=[lo, hi], ci_excludes_zero=excl, rbar=b.get("rbar_ann"),
                 violated=b.get("frac_bind")),
            "Single-curve design: c-bar (compounded annual pp) and its subsampling 95% interval, the "
            "forward rate rbar_f it is evaluated at, and the share of inequalities violated",
            f"{src} sections.tab_bbl_cbar_design.single.blocks.E{E}_{k}", st11,
            run=[dict(routine=rname(E), **{kk: (sc.get("provenance") or {}).get(f"E{E}", {}).get(kk)
                                           for kk in ("tag", "beta", "T", "source")})],
            display=f"{disp(cb)} {disp_ci(lo, hi) if _finite(lo) else '--'}", unit="compounded annual pp",
            checks=[check_cell(T11, f"Panel {k}, c-bar, {rname(E)}",
                               t11.cell(t11.find(r"$\hat{\bar c}", panel=panel[k]), E), [cb]),
                    check_cell(T11, f"Panel {k}, c-bar 95% CI, {rname(E)}",
                               t11.cell(t11.find(r"\quad 95\% CI", panel=panel[k]), E), [lo, hi])])
    scmeds = {}
    for (E, k), b in sblocks.items():
        cm = cmed_entry(f"bbl.t11.cmed.E{E}.{k}", "Table 11", "Cost at median shifters, single curve",
                        b, T11, t11, panel[k], k, E,
                        f"sections.tab_bbl_cbar_design.single.blocks.E{E}_{k}", st11,
                        [dict(routine=rname(E), **{kk: (sc.get("provenance") or {}).get(f"E{E}", {}).get(kk)
                                                   for kk in ("tag", "beta", "T", "source")})])
        if cm:
            scmeds[(E, k)] = cm
    cmed_count("bbl.t11.cmed.count", "Table 11", "Single-curve cost at median shifters", scmeds,
               "sections.tab_bbl_cbar_design.single.blocks", st11)
    if sblocks:
        def _x(b):
            lo, hi = b.get("ci_lo_ann"), b.get("ci_hi_ann")
            return _finite(lo) and _finite(hi) and (lo > 0.0 or hi < 0.0)
        cnt = dict(total=len(sblocks), positive=sum(b["cbar_ann"] > 0.0 for b in sblocks.values()))
        for k in BLOCKS:
            of_k = [b for (E, k2), b in sblocks.items() if k2 == k]
            cnt[f"{k}_blocks"] = len(of_k)
            cnt[f"{k}_excl_zero"] = sum(_x(b) for b in of_k)
        add("bbl.t11.cbar.count", "Table 11", "Single-curve c-bar estimates: signs and intervals, counted",
            cnt, "Over the routine x firm-type blocks of Table 11 (single-curve design): how many c-bar "
            "estimates are positive, and per firm type how many 95% intervals exclude zero",
            f"{src} sections.tab_bbl_cbar_design.single.blocks.*", st11,
            display=f"{cnt['positive']} of {cnt['total']} positive; "
                    + "; ".join(f"{k}: {cnt[f'{k}_excl_zero']} of {cnt[f'{k}_blocks']} exclude 0"
                                for k in BLOCKS))
    return tag


# ── B: phi along the simulated path ──────────────────────────────────────────
PHI_BASES = (("pop", "phi.path", "the simulation's own path (D rows start from the demand "
                                  "prep's stored phi_t)"),
             ("pop_table4", "phi.path_t4", "the same path change on Table 4's base "
                                           "(national_phi_t.csv; no 2016Q1)"))


def collect_phi_path(notes: list):
    """Mean phi-hat along the simulated path at h = 0, 40, 250, on both national bases the summary
    carries, in pp and as a fraction. The footnote's X and Y take one base; which is the user's
    call (they differ by the D rows' national-fill gap reported in the summary's reconciliation)."""
    for E in ROUTINES:
        p = COST_FWD / f"phi_path_summary_E{E}_spec_12.json"
        doc = None
        if p.is_file():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                notes.append(f"{p.name} unreadable ({type(exc).__name__}); treated as pending.")
        hz = (doc or {}).get("horizons") or []
        pooled = (doc or {}).get("pooled") or {}
        for base, stem, what_base in PHI_BASES:
            for h, what in PHI_HORIZONS.items():
                ids = (f"{stem}.E{E}.t{h}", f"{stem}_frac.E{E}.t{h}")
                label = f"Mean phi-hat along the simulated path, {what}, {rname(E)}, {base} base"
                seq = pooled.get(base)
                v = seq[hz.index(h)] if isinstance(seq, list) and h in hz else None
                if doc is None or v is None:
                    for id_, unit in zip(ids, ("pp", "fraction")):
                        add(id_, "B", label, None, f"pooled.{base} at h = {h} ({what_base})",
                            f"{p.name} pooled.{base}[h = {h}]", "pending-phi-summary",
                            display="pending: phi_path_summary", unit=unit)
                    continue
                j = hz.index(h)
                defn = (f"Population-weighted national phi-hat at h = {h} quarters after launch, "
                        f"pooled over launch quarters, on {what_base}. " + (doc.get("definition") or ""))
                src = f"{p.name} pooled.{base}[{j}] (horizons[{j}] = {h})"
                add(ids[0], "B", label, 100.0 * v, defn + " Reported x 100 (pp).", src, "ok",
                    display=f"{disp(100.0 * v)} pp", unit="pp",
                    detail=dict(fraction=v, schema=doc.get("schema"), n_starts=doc.get("n_starts")))
                add(ids[1], "B", label + ", as a fraction", v, defn, src, "ok", display=disp(v),
                    unit="fraction")


# ── C: demand elasticities and the on-impact market elasticity ───────────────
def national_mean_phi(E):
    """Table 4's Mean phi-hat before display: the mean of national_phi_t.csv's spec-12 column
    (sleep_export_spec12_compare.py main)."""
    p = _paths.est_dir(E) / "national_phi_t.csv"
    col = f"phi_t_{_routines.SPEC12_TAG}"
    nd = pd.read_csv(p)
    return float(nd[col].mean()), p, col, int(nd[col].notna().sum())


def add_mean_phi(E, t4):
    """Table 4's Mean phi-hat of routine E, in pp and as a fraction. -> (fraction, pp)."""
    mp, mp_path, mp_col, nq = national_mean_phi(E)
    mp_pp = mp * _st.PHI_DISPLAY
    add(f"sleep.meanphi.E{E}", "C (input)", f"Mean phi-hat (pp), Table 4, {rname(E)}", mp_pp,
        "Population-weighted national phi-hat_t averaged over quarters, in pp (Table 4 row "
        "'Mean phi-hat (pp)')",
        f"{mp_path.relative_to(EST_OUT)} mean of {mp_col} ({nq} quarters) x PHI_DISPLAY",
        "ok", display=disp(mp_pp), unit="pp",
        checks=[check_cell(T4, f"Mean phi-hat row, {rname(E)}",
                           t4.cell(t4.find(r"Mean $\hat{\phi}$"), E), [mp_pp])])
    add(f"sleep.meanphi_frac.E{E}", "C (input)", f"Mean phi-hat as a fraction, {rname(E)}", mp,
        f"sleep.meanphi.E{E} / 100 (the prose's 'phi = 0.983')",
        f"{mp_path.relative_to(EST_OUT)} mean of {mp_col} ({nq} quarters)", "ok",
        display=disp(mp), unit="fraction")
    if mp < 1.0:
        awake_pp = _st.PHI_DISPLAY - mp_pp
        add(f"sleep.awake.E{E}", "C", f"Mean awake share (pp), {rname(E)}", awake_pp,
            "100 minus Table 4's Mean phi-hat (pp): the share of balances re-optimized in a quarter",
            f"100 - sleep.meanphi.E{E}", "ok", display=disp(awake_pp), unit="pp")
        add(f"sleep.awake_frac.E{E}", "C", f"Mean awake share as a fraction, {rname(E)}", 1.0 - mp,
            f"1 - sleep.meanphi_frac.E{E} (the factor of the on-impact elasticity)",
            f"1 - sleep.meanphi_frac.E{E}", "ok", display=disp(1.0 - mp), unit="fraction")
        add(f"sleep.wait.E{E}", "C", f"Average wait until a balance is re-optimized, {rname(E)}",
            dict(quarters=1.0 / (1.0 - mp), years=0.25 / (1.0 - mp)),
            "1 / (1 - mean phi-hat), in quarters and in years: the average wait if phi were "
            "constant at its mean. An illustration of the level of phi-hat, not an estimate",
            f"1 / (1 - sleep.meanphi_frac.E{E})", "ok",
            display=f"{fmt3s(1.0 / (1.0 - mp))} quarters")
    return mp, mp_pp


def collect_demand(notes: list):
    t4, t5, t6 = table(T4), table(T5), table(T6)
    summ = json.loads(LOGIT_SUMMARY.read_text(encoding="utf-8")) if LOGIT_SUMMARY.is_file() else {}
    # The linear routines enter only through this row: their unconstrained link puts the mean
    # above 100, which is what motivates the constrained link of the single-index routines.
    for E in LINEAR_ROUTINES:
        if (_paths.est_dir(E) / "national_phi_t.csv").is_file():
            add_mean_phi(E, t4)
        else:
            notes.append(f"national_phi_t.csv not found for {rname(E)}; sleep.meanphi.E{E} not built.")
    for E in ROUTINES:
        rho = rc.mean_rho_one_minus_s(E)
        mp, mp_pp = add_mean_phi(E, t4)
        ent = summ.get(f"E{E}_{LOGIT_SUBMODEL}") or {}
        names = ent.get("param_names") or []
        a_logit = float(ent["theta1"][names.index("alpha")]) if "alpha" in names else None
        se_logit = (float(ent["se"][names.index("alpha")])
                    if "alpha" in names and len(ent.get("se") or []) == len(names) else None)
        blp = rc.load_stage(E, BLP_STAGE)
        a_blp = rc.alpha_of(blp) if blp else None
        bn = (blp or {}).get("param_names_theta1") or []
        se_blp = (float(blp["theta1_se"][bn.index("alpha")])
                  if "alpha" in bn and len(blp.get("theta1_se") or []) == len(bn) else None)
        g_blp = (blp or {}).get("G_star")
        if _finite(g_blp):
            from scipy import stats as _stats
            add(f"demand.t6.tcrit.E{E}", "C", f"95% Student-t critical value at G*, BLP, {rname(E)}",
                float(_stats.t.ppf(0.975, float(g_blp))),
                "t_{0.975}(G*): the multiple of the standard error that gives the 95% interval "
                "implied by the BLP table's inference (stars from Student-t with G* effective "
                "clusters). The whiskers of the slide coefficient plot",
                f"BLP_RESULTS/cluster_raw/blp_results_E{E}_spec_12_{BLP_STAGE}.json G_star", "ok",
                display=disp(float(_stats.t.ppf(0.975, float(g_blp)))),
                detail=dict(G_star=float(g_blp)))
        ab = {}
        for kind, a, se, tname, src in (
                ("logit", a_logit, se_logit, T5, f"{LOGIT_SUMMARY.relative_to(EST_OUT)} "
                                                 f"E{E}_{LOGIT_SUBMODEL}.theta1/se[alpha]"),
                ("blp", a_blp, se_blp, T6, f"BLP_RESULTS/cluster_raw/blp_results_E{E}_spec_12_"
                                           f"{BLP_STAGE}.json theta1/theta1_se[alpha]")):
            t = t5 if kind == "logit" else t6
            ab[kind] = (a, se)
            est_cells, se_cells = t.find_row("Price coefficient")
            add(f"demand.alpha.{kind}.E{E}", "C (input)",
                f"Price coefficient alpha and its SE, {'Table 5 (logit)' if kind == 'logit' else 'Table 6 (BLP)'}, "
                f"{rname(E)}", dict(estimate=a, se=se),
                "alpha-hat, the average-market price coefficient, and its WCB standard error",
                src, "ok", display=f"{disp(a)} ({disp(se)})",
                checks=[check_cell(tname, f"alpha, {rname(E)}", t.cell(est_cells, E), [a]),
                        check_cell(tname, f"alpha SE, {rname(E)}", t.cell(se_cells, E), [se])])
            el = a * rho if (a is not None and rho is not None) else None
            tab = "Table 5 (logit)" if kind == "logit" else "Table 6 (BLP)"
            add(f"demand.elast.{kind}.E{E}", "C (input)",
                f"Mean own-price elasticity, {tab}, {rname(E)}", el,
                "alpha-hat x mean over the routine's demand rows of rho(1-s), rho = spread_ann/100 "
                "and s = share_B_cond (B rows) or share_D (D rows), 0 <= s < 1: the "
                f"average-market own-price elasticity of the {tab} row",
                f"{src} x mean rho(1-s) of the newest demand_{E}_*spec_12.parquet in DEMAND_PREP",
                "ok", display=disp(el), detail=dict(alpha=a, mean_rho_1ms=rho),
                checks=[check_cell(tname, f"Mean own-price elasticity row, {rname(E)}",
                                   t.cell(t.find("Mean own-price elasticity"), E), [el])])
            imp = el * (1.0 - mp) if el is not None else None
            add(f"demand.impact.{kind}.E{E}", "C",
                f"On-impact market elasticity, {tab.split(' ')[2].strip('()')}, {rname(E)}", imp,
                "The active elasticity times the share of depositors who respond on impact: "
                "elasticity x (1 - mean phi-hat), mean phi-hat as a fraction (Table 4 row / 100)",
                f"demand.elast.{kind}.E{E} x (1 - sleep.meanphi.E{E} / 100)", "ok",
                display=disp(imp),
                detail=dict(formula="elasticity * (1 - mean_phi_pp / 100)", elasticity=el,
                            mean_phi_pp=mp_pp, one_minus_phi=1.0 - mp))
        (a_l, se_l), (a_b, se_b) = ab["logit"], ab["blp"]
        for what, num, den, word in (("alpha", a_b, a_l, "price coefficient"),
                                     ("se", se_b, se_l, "standard error of the price coefficient")):
            r = num / den if _finite(num) and _finite(den) and den != 0 else None
            add(f"demand.ratio.{what}.E{E}", "C", f"BLP {word} over the logit's, {rname(E)}", r,
                f"Table 6's {word} divided by Table 5's (the 'about six times' and 'eleven times' "
                "of the prose)", f"demand.alpha.blp.E{E} / demand.alpha.logit.E{E}, "
                f"{'estimate' if what == 'alpha' else 'se'}", "ok", display=disp(r))


def phi_base_check(notes: list):
    """The path's h = 0 national phi against Table 4's national phi-hat_t, quarter by quarter.
    Both are described as the population-weighted national phi-hat_t; a gap between them is
    reported (in pp) so that X and Y are not read against a Table 4 level on another base."""
    col = f"phi_t_{_routines.SPEC12_TAG}"
    for E in ROUTINES:
        p = COST_FWD / f"phi_path_summary_E{E}_spec_12.json"
        q = _paths.est_dir(E) / "national_phi_t.csv"
        if not (p.is_file() and q.is_file()):
            continue
        try:
            by = json.loads(p.read_text(encoding="utf-8")).get("by_start") or {}
            t0 = {k: float(v["pop"][0]) for k, v in by.items()}
            nd = pd.read_csv(q).set_index("year_quarter")[col]
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            notes.append(f"phi base check E{E} skipped ({type(exc).__name__}: {exc}).")
            continue
        common = [k for k in t0 if k in nd.index]
        if not common:
            continue
        d = np.array([t0[k] - float(nd[k]) for k in common]) * _st.PHI_DISPLAY
        if np.max(np.abs(d)) > 5e-4:
            only = sorted(set(t0) - set(nd.index))
            notes.append(
                f"{rname(E)}: the path's h = 0 national phi ({p.name}, by_start.*.pop[0]) is not "
                f"Table 4's national phi-hat_t ({q.parent.name}/{q.name}, {col}): over the "
                f"{len(common)} common quarters they differ by {fmt3(np.mean(d))} pp on average "
                f"and by up to {fmt3(np.max(np.abs(d)))} pp in one quarter"
                + (f"; launch quarters absent from the csv: {', '.join(only)}" if only else "")
                + ". phi.path.* are on the path's base; phi.path_t4.* carry the same path "
                "change on Table 4's base.")


# ── D: entry-path interval ────────────────────────────────────────────────────
def collect_entry(notes: list):
    p, q = D6_DIR / "d6_phi_in_interval.csv", D6_DIR / "d6_implied_phi.csv"
    iv = pd.read_csv(p)
    pairs = {(float(lo), float(hi)) for lo, hi in zip(iv["lo"], iv["hi"])}
    if len(pairs) != 1:
        notes.append(f"{p.name}: the routines carry different intervals {sorted(pairs)}; the first "
                     "row is reported.")
    lo, hi = float(iv["lo"].iloc[0]), float(iv["hi"].iloc[0])
    add("phi.entry.interval", "D", "Entry-dynamics phi, 95% interval", [lo, hi],
        "95% interval of the sleepy share implied by the B-firm entry paths (d6 entry dynamics, "
        "bootstrap); the band the per-routine phi-hat_mt shares are tested against",
        f"DIAG_PHI_SEPARATION/{p.name} columns lo, hi (identical over the {len(iv)} routine rows)"
        if len(pairs) == 1 else f"DIAG_PHI_SEPARATION/{p.name} columns lo, hi (row estim "
                                f"{int(iv['estim'].iloc[0])})",
        "ok", display=disp_ci(lo, hi))
    if q.is_file():
        ip = pd.read_csv(q)
        b = ip[ip["kind"] == "B"]
        if len(b) == 1:
            v = float(b["phi_entry"].iloc[0])
            same = (abs(float(b["lo"].iloc[0]) - lo) < 1e-12 and abs(float(b["hi"].iloc[0]) - hi) < 1e-12)
            if not same:
                notes.append(f"{q.name} kind B lo/hi differ from {p.name}: the point and the "
                             "interval come from different d6 runs.")
            add("phi.entry.point", "D", "Entry-dynamics phi, point estimate", v,
                "Sleepy share implied by the B-firm entry paths (kind B, median-loss inversion)",
                f"DIAG_PHI_SEPARATION/{q.name} kind B, phi_entry", "ok", display=disp(v),
                detail=dict(n_events=int(b["n_events"].iloc[0]), interval_matches=same))
    m = D6_DIR / "d6_meta.json"
    if m.is_file():
        vint = {str(r[0]): (float(r[1]), str(r[2])) for r in
                json.loads(m.read_text(encoding="utf-8")).get("vintages") or [] if len(r) >= 3}
        for E in ROUTINES:
            if f"E{E}" not in vint:
                continue
            v, prov_ = vint[f"E{E}"]
            add(f"phi.entry.model.E{E}", "D", f"Mean phi-hat behind Figure 2's model curve, {rname(E)}",
                v, "The routine's mean phi_mt the entry-dynamics model curve (III)/(IV) is drawn with; "
                "a different estimand from phi.entry.point (the share implied by the entry paths) and "
                "from Table 4's Mean phi-hat (the national phi_t)",
                f"DIAG_PHI_SEPARATION/{m.name} vintages[E{E}]: {prov_}", "ok", display=disp(v),
                unit="fraction")


# ── F: the discount design and the forward-rate paths ────────────────────────
CF_SIM_JL = pathlib.Path(__file__).resolve().parent / "cf_deposit_sim.jl"


def _stability_kappa():
    """RDEP_STABILITY_KAPPA as cf_deposit_sim.jl sets it: the simulated deposit rate is capped at
    kappa / (beta * phi) - 1."""
    m = re.search(r"^const RDEP_STABILITY_KAPPA\s*=\s*([0-9.]+)", CF_SIM_JL.read_text(
        encoding="utf-8", errors="replace"), re.M)
    return float(m.group(1)) if m else None


def collect_design(notes: list):
    """beta, T and the rate they come from (bbl_discount.env), the discount-weight shares the prose
    quotes, the deposit-rate cap, and each Focus vintage's neutral rate and flat tail
    (forward_rf_vintages.csv). Design constants, the same for every run that reads the registry; a
    table column whose run recorded another beta or T is noted."""
    from utils import bbl_discount as bd
    env = f"{bd.ENV_PATH.name} (utils/bbl_discount.py)"
    b, T, rfq, win = bd.beta(), bd.horizon(), bd.rf_mean_q(), bd.rf_window()
    if rfq is not None:
        add("bbl.design.rf_mean", "F", "Average realized quarterly risk-free rate, launch quarters",
            100.0 * rfq, "BBL_RF_MEAN_Q x 100: the mean over BBL_RF_WINDOW of the realized quarterly "
            "Selic (risk_free_qoq), the rate beta = 1/(1 + rate) is derived from",
            f"{env} BBL_RF_MEAN_Q", "ok", display=f"{fmt3s(100.0 * rfq)}%",
            unit="percent per quarter", detail=dict(window=win, beta_unrounded=1.0 / (1.0 + rfq)))
    add("bbl.design.beta", "F", "Discount factor per quarter", b,
        "BBL_BETA, the per-quarter discount factor every BBL run reads", f"{env} BBL_BETA", "ok",
        display=disp(b))
    add("bbl.design.beta_annual", "F", "Discount factor per year", bd.annual(b),
        "BBL_BETA^4", f"{env} BBL_BETA", "ok", display=disp(bd.annual(b)))
    add("bbl.design.T", "F", "Forward-simulation horizon (quarters)", int(T),
        "BBL_HORIZON: the shortest T whose discount weights cover 99.5% of the infinite-horizon "
        "total", f"{env} BBL_HORIZON", "ok", display=fmt_count(T))
    if win:
        n = bd.quarters_in(*win)
        add("bbl.design.n_launch", "F", "Launch quarters", int(n),
            f"Quarters in BBL_RF_WINDOW ({win[0]}-{win[1]})", f"{env} BBL_RF_WINDOW", "ok",
            display=fmt_count(n))
    cov = bd.coverage(b, T)
    add("bbl.design.coverage", "F", "Share of the discount weight the horizon carries", 100.0 * cov,
        "100 x (1 - beta^T): the first T quarters' share of the infinite-horizon discount weight",
        f"{env} BBL_BETA, BBL_HORIZON", "ok", display=f"{fmt3s(100.0 * cov)}%", unit="percent")
    w40 = (b ** 41 - b ** T) / (1.0 - b ** T)
    add("bbl.design.weight_beyond_q40", "F", "Share of the discount weight beyond quarter 40",
        100.0 * w40, "100 x sum_{t=41}^{T-1} beta^t / sum_{t=0}^{T-1} beta^t, t = 0 the launch "
        "quarter (the simulator's periods): the weight the simulation puts after quarter 40",
        f"{env} BBL_BETA, BBL_HORIZON", "ok", display=f"{fmt3s(100.0 * w40)}%", unit="percent")
    kappa = _stability_kappa()
    phi = national_mean_phi(ROUTINES[0])[0]
    if kappa is None:
        notes.append(f"RDEP_STABILITY_KAPPA not found in {CF_SIM_JL.name}; the cap entry is absent.")
    else:
        cap = kappa / (b * phi) - 1.0
        add("bbl.design.cap", "F", "Deposit-rate cap at the mean sleepy share (percent per quarter)",
            dict(value=100.0 * cap, phi=phi, kappa=kappa),
            f"100 x (kappa / (beta phi) - 1) at phi = Table 4's Mean phi-hat of {rname(ROUTINES[0])} "
            "(as a fraction): the simulated deposit rate's cap per quarter at a typical phi",
            f"{CF_SIM_JL.name} RDEP_STABILITY_KAPPA; {env} BBL_BETA; "
            f"sleep.meanphi_frac.E{ROUTINES[0]}", "ok",
            display=f"{fmt3s(100.0 * cap)}% at phi {fmt3(phi)}", unit="percent per quarter")
    add("bbl.design.sleepy_q40", "F", "Share of the launch sleepy stock left after 40 quarters",
        phi ** 40, f"phi^40 at Table 4's Mean phi-hat of {rname(ROUTINES[0])} (the prose's 'the "
        "inherited sleepy stock has halved' by quarter 40)",
        f"sleep.meanphi_frac.E{ROUTINES[0]}", "ok", display=disp(phi ** 40), unit="fraction")

    p = COST_FWD / "forward_rf_vintages.csv"
    if not p.is_file():
        notes.append(f"{p.name} not found; the neutral-rate and flat-tail entries are absent.")
        return
    fv = pd.read_csv(p)
    neutral = (fv.groupby("start_q")["source"].first().str.extract(r"neutral:([-\d.]+)")[0]
               .astype(float))
    rows = []
    for q, g in fv.groupby("start_q"):
        g = g.sort_values("h")
        lev, hh = g["selic_ann_pct"].to_numpy(float), g["h"].to_numpy(int)
        k = len(lev)
        while k > 0 and abs(lev[k - 1] - lev[-1]) < 1e-4:
            k -= 1
        t = hh[hh <= T - 1]
        w = b ** t.astype(float)
        rows.append(dict(q=q, h_flat=int(hh[k]), share=float(w[t >= hh[k]].sum() / w.sum()),
                         terminal=float(lev[-1]), covers=bool(hh.max() >= T - 1)))
    r = pd.DataFrame(rows)
    if not r["covers"].all():
        notes.append(f"{p.name}: some vintages stop before h = T - 1 = {T - 1}; their flat-tail "
                     "shares are over the rows present.")
    bad = r[(r["terminal"] - neutral.reindex(r["q"]).to_numpy()).abs() > 5e-3]
    if len(bad):
        notes.append(f"{p.name}: the curve's terminal level differs from its 'neutral:' tag in "
                     f"{', '.join(bad['q'])}.")
    lo, hi = float(neutral.min()), float(neutral.max())
    add("bbl.rate.neutral", "F", "Expected neutral rate across Focus vintages (range)", [lo, hi],
        "Min and max over the launch quarters' Focus vintages of the median expectation for the "
        "last reference year, at which each path is held flat (compounded annual percent)",
        f"COST_FWD/{p.name} source 'neutral:' tag (= the curve's terminal selic_ann_pct)", "ok",
        display=f"{disp_ci(lo, hi)} %", unit="percent per year",
        detail=dict(min_at=sorted(neutral[neutral == lo].index),
                    max_at=sorted(neutral[neutral == hi].index), n_vintages=int(len(neutral))))
    add("bbl.rate.flat_weight", "F", "Discount weight on each vintage's flat tail",
        dict(min=100.0 * r["share"].min(), max=100.0 * r["share"].max(),
             mean=100.0 * r["share"].mean()),
        "Per launch quarter, 100 x sum_{t >= h_flat} beta^t / sum_{t=0}^{T-1} beta^t, t the "
        "simulator's period (rate path h = t) and h_flat the first quarter from which the curve "
        "stays at its terminal (neutral) level; min, max and mean over the launch quarters",
        f"COST_FWD/{p.name} selic_ann_pct by start_q and h; {env} BBL_BETA, BBL_HORIZON", "ok",
        display=f"{fmt3s(100 * r['share'].min())}-{fmt3s(100 * r['share'].max())}% "
                f"(mean {fmt3s(100 * r['share'].mean())}%)", unit="percent",
        detail=dict(min_at=r.loc[r["share"].idxmin(), "q"], max_at=r.loc[r["share"].idxmax(), "q"],
                    h_flat={row.q: row.h_flat for row in r.itertuples()}))
    # The runs the tables quote must have been solved with the registry's beta and T.
    for e in ENTRIES:
        if e["id"].startswith(("bbl.t9.col.", "bbl.t11.col.")) and e["status"] == "ok":
            v = e["value_full"] or {}
            if v.get("beta") is not None and (abs(float(v["beta"]) - b) > 1e-9 or int(v["T"]) != T):
                notes.append(f"{e['id']}: the run's beta {v['beta']} / T {v['T']} are not the "
                             f"registry's {b} / {T}; the design entries do not describe it.")


# ── G: the table rows the prose quotes (Tables 4-6, as rendered) ─────────────
T4_ROWS = (("Constant", "const"), ("Cad", "cadunico"), ("Fraction 65+", "age65"),
           ("Lagged Selic", "selic"), ("Mobile Lines", "mobile"), ("Pix Available", "pix"),
           ("GDP Growth", "gdp_growth"))
T56_ROWS = (("Price coefficient", "alpha"), ("FGC Covered", "fgc"), ("Group Contains IP", "ip"),
            (r"$\ln(\text{Total Assets}", "lnassets"), ("State-Owned", "state_owned"),
            (r"$\Sigma$(Spread)", "sigma_spread"), (r"$\Pi$(Spread $\times$ GDP", "pi_spread_gdp"),
            (r"$\Pi$(Spread $\times$ Fraction 65+", "pi_spread_age65"),
            (r"$\Pi$(Spread $\times$ Mobile Lines", "pi_spread_mobile"),
            (r"$\Pi$($\ln$ Assets $\times$ GDP", "pi_assets_gdp"))
T4_STATS = (("Observations", "n_obs", True), ("Clusters", "n_clusters", True),
            ("$R^2$", "r2", False))
T56_STATS = (("Observations", "n_obs", True), ("GMM Criterion", "q", False),
             ("Effective Clusters", "gstar", False))
SIG_LEVEL = {3: "1", 2: "5", 1: "10", 0: "none"}


T4_SIDECAR = _paths.rout_dir() / "est1-4_spec12_stage2_comparison.json"
T4_VARS = {"nr_lagged_dep": "const", "interaction_cadunico_families_per1000": "cadunico",
           "interaction_fraction_65plus": "age65", "interaction_risk_free_qoq_lag": "selic",
           "interaction_connections_per100": "mobile", "interaction_pix_exists": "pix",
           "interaction_gdp_growth_yoy": "gdp_growth"}


def collect_table4_sidecar(notes: list) -> dict:
    """Table 4's rows from the sidecar its generator writes next to its fits
    (sleep_export_spec12_compare.py): unrounded estimates, 95% intervals, the stars as printed and
    the dagger of the quarter-clustered rows, each checked against the rendered cell.
    -> {E: Pix AME} for the awake-share entry; None when the sidecar is absent or not Table 4's."""
    if not T4_SIDECAR.is_file():
        notes.append(f"{T4_SIDECAR.name} not found in Rout; Table 4's rows are read from the "
                     "rendered table at three decimals (rounded3).")
        return None
    sc = json.loads(T4_SIDECAR.read_text(encoding="utf-8"))
    if sc.get("tex_file") != T4:
        notes.append(f"{T4_SIDECAR.name} describes {sc.get('tex_file')}, not {T4}; not used.")
        return None
    t4 = table(T4)
    est_of = sc.get("column_est_id") or {}
    src = f"Rout/{T4_SIDECAR.name} (written with {T4} by sleep_export_spec12_compare.py)"
    pix = {}
    for r in sc.get("rows") or []:
        slug = T4_VARS.get(r["var"], r["var"].removeprefix("interaction_"))
        label = r.get("row_label") or r["var"]
        cells, sub = t4.find_row(label.split(" (")[0])
        for col, c in (r.get("cells") or {}).items():
            E = est_of.get(col)
            if E is None or not _finite(c.get("estimate")):
                continue
            stars = min(len(c.get("stars") or ""), 3)
            v = dict(estimate=c["estimate"], ci=[c.get("ci_lo"), c.get("ci_hi")], stars=stars,
                     sig_level=SIG_LEVEL[stars], quarter_clustered=bool(c.get("dagger")))
            if slug == "pix":
                pix[E] = c["estimate"]
            add(f"sleep.t4.{slug}.E{E}", "G", f"Table 4, {label}, {rname(E)}", v,
                f"Table 4 row '{label}', column {rname(E)}: the AME (pp of the sleepy share), its "
                "95% interval (quarter_clustered: marked with a dagger), the stars and the "
                "significance level they mark (1, 5, 10 percent, or none)",
                f"{src} rows[{r['var']}].cells[{col}]", "ok",
                display=f"{fmt3s(c['estimate'])}{'*' * stars} {disp_ci(c.get('ci_lo'), c.get('ci_hi'))}",
                checks=[check_cell(T4, f"{label}, {rname(E)}", t4.cell(cells, E), [c["estimate"]]),
                        check_cell(T4, f"{label} 95% interval, {rname(E)}", t4.cell(sub, E),
                                   [c.get("ci_lo"), c.get("ci_hi")])])
    tab = sc.get("table") or {}
    for key, slug, rowlab, is_count in (("observations", "n_obs", "Observations", True),
                                        ("r_squared", "r2", "$R^2$", False),
                                        ("clusters", "n_clusters", "Clusters", True)):
        cells, _ = t4.find_row(rowlab)
        for col, c in (tab.get(key) or {}).items():
            E = est_of.get(col)
            raw = c.get("value", c.get("unrounded"))
            if E is None or not _finite(raw):
                continue
            v = int(round(raw)) if is_count else float(raw)
            add(f"sleep.t4.{slug}.E{E}", "G", f"Table 4, {rowlab}, {rname(E)}", v,
                f"Table 4 row '{rowlab}', column {rname(E)}", f"{src} table.{key}[{col}]", "ok",
                display=fmt_count(v) if is_count else fmt3s(v),
                checks=[check_cell(T4, f"{rowlab}, {rname(E)}", t4.cell(cells, E), [v])])
    return pix


def collect_table_rows(notes: list):
    """Each quoted coefficient row of Tables 4-6 per routine column: the estimate, its interval
    (Table 4) or standard error (Tables 5-6), the stars and the significance level they mark, plus
    the sample rows. Table 4 comes unrounded from its sidecar (collect_table4_sidecar); Tables 5-6,
    and Table 4 when the sidecar is absent, are read from the rendered table at three decimals
    (rounded3), since their generators export no unrounded values for these rows. Also the Pix
    effect relative to the awake share."""
    sc_pix = collect_table4_sidecar(notes)
    pix = dict(sc_pix or {})
    for tname, stem, tlab, rows, stats, second in (
            (T4, "sleep.t4", "Table 4", T4_ROWS, T4_STATS, "ci"),
            (T5, "demand.t5", "Table 5 (logit)", T56_ROWS, T56_STATS, "se"),
            (T6, "demand.t6", "Table 6 (BLP)", T56_ROWS, T56_STATS, "se")):
        if tname == T4 and sc_pix is not None:
            continue
        t = table(tname)
        if not t.ok:
            notes.append(f"{tname} not rendered; its row entries (G) are absent.")
            continue
        for prefix, slug in rows:
            cells, sub = t.find_row(prefix)
            if cells is None:
                continue
            for E in t.cols:
                txt = t.cell(cells, E)
                nums = cell_numbers(txt) if txt else []
                if not isinstance(E, int) or not nums:
                    continue
                stars = min(txt.count("*"), 3)
                v = dict(estimate=nums[0], stars=stars, sig_level=SIG_LEVEL[stars])
                below = t.cell(sub, E) if sub else None
                bn = cell_numbers(below) if below else []
                if second == "ci" and len(bn) >= 2:
                    v["ci"] = [bn[0], bn[1]]
                    v["quarter_clustered"] = "dagger" in below
                elif second == "se" and bn:
                    v["se"] = bn[0]
                if slug == "pix" and tname == T4:
                    pix[E] = nums[0]
                shown = (disp_ci(*v["ci"]) if "ci" in v else
                         f"({fmt3s(v['se'])})" if "se" in v else "")
                add(f"{stem}.{slug}.E{E}", "G", f"{tlab}, {prefix}, {rname(E)}", v,
                    f"{tlab} row '{prefix}', column {rname(E)}: estimate, "
                    + ("95% interval (quarter_clustered: marked with a dagger)" if second == "ci"
                       else "standard error")
                    + ", stars and the significance level they mark (1, 5, 10 percent, or none)",
                    f"rendered {tname} (3 dp; the generator exports no unrounded values)", "ok",
                    rounded3=True, display=f"{fmt3s(nums[0])}{'*' * stars} {shown}".strip())
        for prefix, slug, is_count in stats:
            cells, _ = t.find_row(prefix)
            if cells is None:
                continue
            for E in t.cols:
                txt = t.cell(cells, E)
                nums = cell_numbers(txt) if txt else []
                if not isinstance(E, int) or not nums:
                    continue
                v = int(round(nums[0])) if is_count else nums[0]
                add(f"{stem}.{slug}.E{E}", "G", f"{tlab}, {prefix}, {rname(E)}", v,
                    f"{tlab} row '{prefix}', column {rname(E)}", f"rendered {tname}", "ok",
                    rounded3=not is_count, display=fmt_count(v) if is_count else fmt3s(v))
    for E in ROUTINES:
        if E not in pix:
            continue
        mp_pp = national_mean_phi(E)[0] * _st.PHI_DISPLAY
        rel = 100.0 * (-pix[E]) / (100.0 - mp_pp)
        add(f"sleep.pix.awake_rel.E{E}", "G",
            f"Pix effect relative to the average awake share, {rname(E)}", rel,
            "100 x (minus the Pix Available AME of Table 4, pp of the sleepy share) / (100 - Mean "
            "phi-hat in pp): the percent by which the Pix shift raises the average awake share (the "
            "'roughly forty percent' of the prose)", f"sleep.t4.pix.E{E}.estimate; sleep.meanphi.E{E}",
            "ok", display=f"{fmt3s(rel)}%", unit="percent",
            detail=dict(pix_ame_pp=pix[E], awake_share_pp=100.0 - mp_pp))


# ── H: descriptives (Tables 2 and 3) ──────────────────────────────────────────
T2_VARS = (("log_total_assets", "Log Total Assets", True), ("log_dep", "Log Deposits", True),
           ("log_dep_a4", "Log Deposits (4)", True), ("spread_ann_a4", "Spread (4)", False),
           ("equity_ratio_pct", "Equity Ratio", False))


def _stars_p(p) -> int:
    return 0 if not _finite(p) else 3 if p < 0.01 else 2 if p < 0.05 else 1 if p < 0.10 else 0


def collect_descriptives(notes: list):
    """Table 2 (D firms before and after Pix) from the CSV make_desc_compressed_tables.py writes
    with it, and Table 3 (deposit concentration) plus the sleepiness sample's effective clusters
    from the JSON sleep_desc_clusters.py writes with it; each checked against the rendered table."""
    rout = _paths.rout_dir()
    cands = [p for p in rout.glob("Compressed_DFirm_PrePost_Pix_*.csv")]
    t2 = table(T2)
    if not cands:
        notes.append("Compressed_DFirm_PrePost_Pix_*.csv not found in Rout; Table 2 entries absent.")
    else:
        p = max(cands, key=lambda q: q.stat().st_mtime)
        df = pd.read_csv(p).set_index("var")
        for var, lab, is_log in T2_VARS:
            if var not in df.index:
                continue
            r = df.loc[var]
            stars = _stars_p(r["p"])
            v = dict(pre=float(r["pre_mean"]), post=float(r["post_mean"]), diff=float(r["diff"]),
                     se=float(r["se"]), p=float(r["p"]), stars=stars, sig_level=SIG_LEVEL[stars],
                     G=int(r["G"]), G_star=float(r["G_star"]), n_pre=int(r["n_pre"]),
                     n_post=int(r["n_post"]))
            if is_log:
                v["pct_change"] = 100.0 * math.expm1(float(r["diff"]))
                v["ratio"] = math.exp(float(r["diff"]))
            cells, _ = t2.find_row(lab, exact=(var == "log_dep"))
            add(f"desc.t2.{var}", "H", f"Table 2, {lab}, D firms before and after Pix", v,
                "Pre- and post-Pix means over D conglomerate-quarters (post: from 2020Q4), their "
                "difference (post minus pre) with its conglomerate-clustered SE, the p-value from "
                "t(G*) and the stars it earns, G and G*"
                + ("; pct_change = 100 x (exp(diff) - 1) and ratio = exp(diff), the log difference "
                   "read as a percent change and as a multiple" if is_log else ""),
                f"Rout/{p.name} row var={var}", "ok",
                display=f"{fmt3s(v['diff'])}{'*' * stars} ({fmt3s(v['se'])})",
                checks=[check_cell(T2, f"{lab} pre mean", cells[0] if cells else None, [v["pre"]]),
                        check_cell(T2, f"{lab} post mean", cells[1] if cells and len(cells) > 1 else None,
                                   [v["post"]]),
                        check_cell(T2, f"{lab} difference", cells[2] if cells and len(cells) > 2 else None,
                                   [v["diff"]])])
    cells, _ = t2.find_row("Active D Conglomerates")
    nums = [cell_numbers(c) for c in (cells or [])[:3]]
    if len(nums) == 3 and all(nums):
        pre, post = int(nums[0][0]), int(nums[1][0])
        add("desc.t2.active_d", "H", "Table 2, active D conglomerates before and after Pix",
            dict(pre=pre, post=post, change=post - pre, ratio=post / pre),
            "Count of active D conglomerates before and after Pix and its ratio (the prose's "
            "'doubles')", f"rendered {T2}, row 'Active D Conglomerates' (integers)", "ok",
            display=f"{pre} -> {post} (x{fmt3s(post / pre)})")

    collect_market_structure(notes)
    collect_table1_spreads(notes)
    collect_mca_count(notes)
    collect_slide_deposit_types(notes)

    j = rout / "cluster_imbalance.json"
    if not j.is_file():
        notes.append("Rout/cluster_imbalance.json not found; Table 3 and sleep G* entries absent.")
        return
    ci = json.loads(j.read_text(encoding="utf-8"))
    top = ci.get("top_n_by_deposit_share") or []
    t3 = table(T3)
    if top:
        last = top[-1]
        cells, _ = t3.find_row(f"{len(top)}\\textsuperscript")
        add("desc.t3.top_share", "H", f"Table 3, deposit share of the {len(top)} largest conglomerates",
            dict(cumulative=100.0 * last["cum_share"], n=len(top), obs_share=100.0 * sum(
                r["obs_share"] for r in top)),
            f"Cumulative national deposit share of the {len(top)} largest conglomerates (percent) and "
            "their share of estimation observations, spec-12 second-stage sample",
            f"Rout/{j.name} top_n_by_deposit_share[{len(top) - 1}].cum_share", "ok",
            display=f"{fmt3s(100.0 * last['cum_share'])}%", unit="percent",
            checks=[check_cell(T3, f"{len(top)}th row, cumulative share",
                               cells[1] if cells and len(cells) > 1 else None,
                               [100.0 * last["cum_share"]])])
    add("sleep.clusters", "H", f"Sleepiness sample clusters, E{ci.get('est')} spec-12 second stage",
        dict(G=int(ci["G_nominal"]), G_star=float(ci["G_star"]), cv=float(ci["coefficient_variation"]),
             n_obs=int(ci["total_observations"])),
        "Nominal conglomerate clusters G, effective clusters G* = G/(1+CV^2) (Carter, Schnepel and "
        "Steigerwald 2017) and the coefficient of variation of cluster sizes, in the sleepiness "
        "estimation sample (the 'about six effective ones' of the prose)",
        f"Rout/{j.name} G_nominal, G_star, coefficient_variation", "ok",
        display=f"G {ci['G_nominal']}, G* {fmt3s(ci['G_star'])}")


# Table A.4's columns: (CSV var, id slug, label, rendered panel, column in it, printed as integer).
TA4 = "Compressed_MarketStructure_by_Year_AB.tex"
TA4_VARS = (
    ("n_b_firms", "n_b", "B firms per MCA (mean)", "Panel A", 0, False),
    ("hhi_b", "hhi_b", "HHI of B deposits within the MCA", "Panel A", 1, True),
    ("n_d_firms_natl", "n_d", "D firms, national count", "Panel A", 2, True),
    ("hhi_combined_natl", "hhi_bd", "National HHI, B and D pooled", "Panel A", 3, True),
    ("risk_free_ann", "selic", "Selic (% p.a.)", "Panel B", 0, False),
    ("spread_ann_a4_w", "spread4.B", "Time-deposit spread, B firms (pp p.a.)", "Panel B", 1, False),
    ("spread_ann_a4_d_w", "spread4.D", "Time-deposit spread, D firms (pp p.a.)", "Panel B", 2, False),
    ("spread_ann_a5_w", "spread5.B", "Prepaid spread, B firms (pp p.a.)", "Panel B", 3, False),
    ("spread_ann_a5_d_w", "spread5.D", "Prepaid spread, D firms (pp p.a.)", "Panel B", 4, False),
)
T1 = "Compressed_BankType_CrossSection.tex"
TVARS = "Appendix_Variables_Master.tex"


def collect_market_structure(notes: list):
    """Table A.4 (market structure, the Selic and the spreads by year) from the CSV
    make_desc_compressed_tables.py writes with it: one entry per column, one leaf per year."""
    p = _paths.rout_dir() / "Compressed_MarketStructure_by_Year_dep_weighted.csv"
    if not p.is_file():
        notes.append(f"{p.name} not found in Rout; Table A.4 entries absent.")
        return
    df = pd.read_csv(p).set_index("var")
    ta4 = table(TA4)
    years = [c for c in df.columns]
    for var, slug, lab, panel, col, is_int in TA4_VARS:
        if var not in df.index:
            continue
        v, checks = {}, []
        for y in years:
            x = pd.to_numeric(df.loc[var, y], errors="coerce")
            if not _finite(float(x)):
                continue
            v[str(y)] = float(x)
            cells = ta4.find(str(y), panel=panel, exact=True)
            checks.append(check_cell(TA4, f"{lab}, {y}", cells[col] if cells and col < len(cells) else None,
                                     [float(f"{float(x):.0f}") if is_int else float(x)]))
        if not v:
            continue
        ys = sorted(v)
        add(f"desc.a4.{slug}", "H", f"Table A.4, {lab}, by year", v,
            "Annual mean of the quarterly values, as Table A.4 prints it (deposit-weighted across "
            "MCA-quarters for the B columns; national for the D columns and the Selic); one leaf "
            "per year, e.g. " + f"desc.a4.{slug}.{ys[0]}",
            f"Rout/{p.name} row var={var}", "ok",
            display=f"{ys[0]}: {fmt3s(v[ys[0]])} -> {ys[-1]}: {fmt3s(v[ys[-1]])}", checks=checks)


def collect_table1_spreads(notes: list):
    """Table 1's spread rows (conglomerate-quarter moments, B and D) from the CSVs
    make_desc_compressed_tables.py writes with it."""
    t1 = table(T1)
    for nth, k in enumerate(BLOCKS):
        p = _paths.rout_dir() / f"Compressed_BankType_CrossSection_{k}_unweighted.csv"
        if not p.is_file():
            notes.append(f"{p.name} not found in Rout; Table 1 spread entries for {k} absent.")
            continue
        df = pd.read_csv(p).set_index("var")
        for a in (4, 5):
            var = f"spread_ann_a{a}"
            if var not in df.index:
                continue
            r = df.loc[var]
            v = dict(mean=float(r["Mean"]), sd=float(r["SD"]), median=float(r["p50"]), n=int(r["N"]))
            cells = t1.find(f"Spread ({a})", nth=nth)
            cell = lambda i: cells[i] if cells and i < len(cells) else None
            add(f"desc.t1.spread{a}.{k}", "H", f"Table 1, spread of type {a}, {k} firms", v,
                "Mean, standard deviation and median over conglomerate-quarters of the "
                "deposit-weighted annualized spread (pp p.a.), and the number of them",
                f"Rout/{p.name} row var={var}", "ok", display=f"{fmt3s(v['mean'])} ({fmt3s(v['sd'])})",
                unit="pp p.a.",
                checks=[check_cell(T1, f"Spread ({a}) mean, {k}", cell(0), [v["mean"]]),
                        check_cell(T1, f"Spread ({a}) SD, {k}", cell(1), [v["sd"]]),
                        check_cell(T1, f"Spread ({a}) median, {k}", cell(4), [v["median"]]),
                        check_cell(T1, f"Spread ({a}) N, {k}", cell(7), [v["n"]])])


def collect_mca_count(notes: list):
    """The number of local markets: distinct MCAs with a B-firm row in the market panel inside
    the estimation window, the count the variables table prints."""
    p = EST_OUT.parent / "market_panel.parquet"
    if not p.is_file():
        notes.append(f"{p.name} not found; the MCA count entry is absent.")
        return
    df = pd.read_parquet(p, columns=["year", "mca_code"])
    df = df[df["mca_code"].astype(str) != "NATIONAL"]
    n_all = int(df["mca_code"].nunique())
    n_win = int(df.loc[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR), "mca_code"].nunique())
    f = TABLES_DIR / TVARS
    m = re.search(r"one of ([\d,]+) MCAs", f.read_text(encoding="utf-8", errors="replace")) \
        if f.is_file() else None
    printed = int(m.group(1).replace(",", "")) if m else None
    add("desc.n_mcas", "H", "Number of local markets (MCAs)", n_win,
        f"Distinct MCAs with a B-firm row in the market panel inside {MIN_YEAR}-{MAX_YEAR}, the "
        "sample every table describes",
        f"{p.name}, distinct mca_code other than NATIONAL", "ok", display=fmt_count(n_win),
        detail=dict(all_panel_years=n_all),
        checks=[dict(table=TVARS, cell="'one of N MCAs'", expected=[n_win], printed=[printed],
                     raw=m.group(0) if m else None, ok=printed == n_win,
                     why=None if printed == n_win else ("text not found" if m is None else "differs"))])


def collect_slide_deposit_types(notes: list):
    """The deposit-types table of the slides, from the sidecar make_slide_figures.py writes with
    it: per deposit type, the mean spread and the share of balances, each checked against the
    rendered table. These are slide numbers, not the paper's: the paper prints the spreads by
    year (Table A.4) and by bank (Table 1), and no balance shares."""
    p = _paths.rout_dir() / "slide_deposit_types.json"
    if not p.is_file():
        notes.append(f"{p.name} not found in Rout; run make_slide_figures.py --only table for the "
                     "slide.types.* entries.")
        return
    sc = json.loads(p.read_text(encoding="utf-8"))
    t = table(sc.get("tex_file") or "tab_slide_deposit_types.tex")
    y0, y1 = sc.get("window") or [MIN_YEAR, MAX_YEAR]

    def chk(where, cell, v):
        nums = cell_numbers(cell) if cell else []
        got = nums[-1] if nums else None
        ok = got is not None and fmt3(got) == fmt3(v)
        return dict(table=t.name, cell=where, expected=[fmt3(v)],
                    printed=None if got is None else [fmt3(got)], raw=cell, ok=ok,
                    why=None if ok else ("cell not found" if got is None else "differs"))

    for slug, r in (sc.get("types") or {}).items():
        lab, sp = r["label"], r["spread"]
        cells = t.find(lab)
        cell = lambda i: cells[i] if cells and -len(cells) <= i < len(cells) else None
        if isinstance(sp, dict):
            checks = [chk(f"{lab} spread, {k} firms", cell(1 + i), sp[k]) for i, k in enumerate(BLOCKS)]
            shown = " / ".join(f"{k} {fmt3s(sp[k])}" for k in BLOCKS)
        else:
            checks = [chk(f"{lab} spread", cell(1), sp)]
            shown = fmt3s(sp)
        checks.append(chk(f"{lab} share of balances", cell(-1), r["share"]))
        add(f"slide.types.{slug}", "slides", f"Deposit-types slide table, {lab}",
            dict(spread=sp, share=r["share"]),
            f"`spread`: mean over {y0}-{y1} of Selic minus the deposit rate, compounded annual pp "
            "(regulated types: from the quarterly Selic and savings rate; types 4 and 5: the mean "
            "of Table A.4's annual balance-weighted spreads, B and D). `share`: the type's share "
            f"of the four types' balances, B and D pooled, mean over the quarters of {y0}-{y1}, "
            "in percent",
            f"Rout/{p.name} types.{slug} (written with {t.name} by make_slide_figures.py)", "ok",
            display=f"spread {shown}; share {fmt3s(r['share'])}%", checks=checks)


# ── K: strength of the sleepiness first stage ────────────────────────────────
FS_SPEC = "IV_HausmanFull x Tech"   # specification 12
FS_ROW_EFF = r"Effective $F$ (excluded instruments)"
FS_ROW_ALL = r"$F$, all slopes"


def _tech_panel_cell(tname: str, label: str):
    """Last cell (the Hausman column) of the row `label` in the Tech panel of a first-stage table;
    None when the table or the row is absent."""
    path = TABLES_DIR / tname
    if not path.is_file():
        return None
    in_tech, cell = False, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith(r"\multicolumn") and "Panel" in s:
            in_tech = "Tech" in s
        elif in_tech and s.startswith(label):
            cell = re.sub(r"\\\\\*?\s*$", "", s).split("&")[-1].strip()
    return cell


def collect_first_stage_strength(notes: list):
    """The first stage of the sleepiness estimation under specification 12, per routine, from the
    sidecar sleep_first_stage_strength.py writes: the effective F of the excluded instruments
    (Montiel Olea and Pflueger), their joint F, the joint F of all slopes, and the critical values.
    The two F rows the first-stage tables print are checked against it."""
    p = _paths.rout_dir() / "sleep_first_stage_strength.json"
    if not p.is_file():
        notes.append(f"{p.name} not found in Rout; run sleep_first_stage_strength.py for the "
                     "sleep.fs.* entries.")
        return
    sc = json.loads(p.read_text(encoding="utf-8"))
    ident = (sc.get("_meta") or {}).get("identity_check") or {}
    if ident.get("passed") is not True:
        notes.append(f"{p.name}: its identity check against weak_iv_sleep.json is not recorded as "
                     "passed; the sleep.fs.* entries are not built.")
        return
    src = f"Rout/{p.name} (written by sleep_first_stage_strength.py from the stored fits)"
    effs = {}
    for E in _routines.ACTIVE:
        r = (sc.get(f"E{E}") or {}).get(FS_SPEC)
        if not r or not _finite(r.get("eff_F")):
            notes.append(f"{p.name} has no {FS_SPEC} cell for {rname(E)}.")
            continue
        tname = f"est{E}_first_stage_table.tex"
        eff, excl, fall = r["eff_F"], r.get("excl_F"), r.get("F_all_slopes")
        cv = r.get("mop_cv") or {}
        effs[E] = eff
        add(f"sleep.fs.eff_f.E{E}", "sleep first stage", f"Effective F, first stage, {rname(E)}",
            eff, "Effective F of the seven excluded instruments of specification 12 (Montiel Olea "
            "and Pflueger), conglomerate-clustered, on the first stage as estimated (types 4 and 5 "
            "pooled, constant and state controls partialled out)",
            f"{src} E{E}.{FS_SPEC}.eff_F", "ok", display=disp(eff),
            detail=dict(N=r.get("N"), G=r.get("G"), n_iv=r.get("n_iv")),
            checks=[check_cell(tname, f"Tech panel, {FS_ROW_EFF}, Hausman column",
                               _tech_panel_cell(tname, FS_ROW_EFF), [eff])])
        if _finite(excl):
            add(f"sleep.fs.excl_f.E{E}", "sleep first stage",
                f"F of the excluded instruments, first stage, {rname(E)}", excl,
                "Cluster-robust joint F of the seven excluded instruments of specification 12 "
                "(not printed in the tables)",
                f"{src} E{E}.{FS_SPEC}.excl_F", "ok", display=disp(excl))
        if _finite(fall):
            add(f"sleep.fs.f_all.E{E}", "sleep first stage",
                f"F of all slopes, first stage, {rname(E)}", fall,
                "Joint F of every slope of the first-stage regression, instruments and state "
                "controls together (the row '$F$, all slopes' of the first-stage table)",
                f"{src} E{E}.{FS_SPEC}.F_all_slopes", "ok", display=disp(fall),
                checks=[check_cell(tname, f"Tech panel, {FS_ROW_ALL}, Hausman column",
                                   _tech_panel_cell(tname, FS_ROW_ALL), [fall])])
        if all(_finite(cv.get(k)) for k in ("bias10", "bias20")):
            add(f"sleep.fs.cv.E{E}", "sleep first stage",
                f"Critical values of the effective F, {rname(E)}",
                dict(bias05=cv.get("bias05"), bias10=cv["bias10"], bias20=cv["bias20"],
                     bias30=cv.get("bias30")),
                "Montiel Olea and Pflueger critical values (simplified procedure, Nagar bias of "
                "TSLS, 5% level) for a worst-case bias of 5, 10, 20 and 30 percent",
                f"{src} E{E}.{FS_SPEC}.mop_cv", "ok",
                display=f"10%: {disp(cv['bias10'])}; 20%: {disp(cv['bias20'])}")
    if effs:
        lo, hi = min(effs.values()), max(effs.values())
        add("sleep.fs.eff_f.range", "sleep first stage",
            "Effective F, first stage, range over routines", [lo, hi],
            "min and max over the routines of the effective F under specification 12",
            f"{src} E*.{FS_SPEC}.eff_F", "ok", display=disp_ci(lo, hi))


# ── I: the policy-function fit (Tables C.10 and C.11) ────────────────────────
POLFUNC_COLS =(("B", "B"), ("D_optA", "D, national means"), ("D_optB", "D, with demographics"))
POLFUNC_K = ((4, "k4_Time_CDB", TC10, 100.0, "pp"), (5, "k5_Prepaid", TC11, 1e4, "bp"))


def collect_polfunc(notes: list):
    """R^2, mean of the dependent variable (the mean fitted spread, equal to it in-sample), N, G and
    G* of each policy-function column, from polfunc_summary.json, checked against C.10/C.11."""
    p = _paths.polfunc_dir() / "polfunc_summary.json"
    if not p.is_file():
        notes.append(f"{p.name} not found; policy-function entries absent.")
        return
    sm = json.loads(p.read_text(encoding="utf-8"))
    for k, stem, tname, scale, unit in POLFUNC_K:
        t = table(tname)
        rows = {lab: t.find_row(lab)[0] for lab in ("$R^{2}$", "Mean dep.", "Observations",
                                                    "Clusters", "Eff.")}
        r2s = {}
        for j, (col, word) in enumerate(POLFUNC_COLS):
            s = sm.get(f"{stem}_{col}")
            if not s:
                continue
            r2s[col] = float(s["r_squared"])
            v = dict(r2=float(s["r_squared"]), mean_spread=scale * float(s["mean_depvar"]),
                     n_obs=int(s["n_obs"]), G=int(s["n_clusters"]), G_star=float(s["G_star"]))

            def cell(lab):
                c = rows.get(lab)
                return c[j] if c and len(c) > j else None
            add(f"polfunc.k{k}.{col}", "I", f"Table C.1{0 if k == 4 else 1} column {word}: fit", v,
                f"R^2, mean spread ({unit}; the mean of the dependent variable, which the fitted "
                "values share in-sample), observations, clusters and effective clusters of the "
                f"k = {k} policy function, {word}",
                f"COST_POLFUNC/{p.name} {stem}_{col}", "ok",
                display=f"R2 {fmt3s(v['r2'])}; mean {fmt3s(v['mean_spread'])} {unit}",
                checks=[check_cell(tname, f"{word} R^2", cell("$R^{2}$"), [v["r2"]]),
                        check_cell(tname, f"{word} mean dep. var.", cell("Mean dep."), [v["mean_spread"]]),
                        check_cell(tname, f"{word} observations", cell("Observations"), [v["n_obs"]]),
                        check_cell(tname, f"{word} G*", cell("Eff."), [v["G_star"]])])
        d = [r2s[c] for c in ("D_optA", "D_optB") if c in r2s]
        if d:
            add(f"polfunc.k{k}.D_r2", "I", f"Table C.1{0 if k == 4 else 1}: R^2 of the D columns",
                dict(lo=min(d), hi=max(d), mean=float(np.mean(d))),
                "Min, max and mean R^2 over the two D columns (national means; with demographics)",
                f"COST_POLFUNC/{p.name} {stem}_D_opt{{A,B}}.r_squared", "ok",
                display=f"{fmt3s(min(d))}-{fmt3s(max(d))}")


# ── J: the entry-event counts behind Figure 2 ────────────────────────────────
def collect_entry_counts(notes: list):
    p = D6_DIR / "d6_entry_paths.csv"
    if not p.is_file():
        return
    ep = pd.read_csv(p)
    h0 = ep[ep["h"] == ep["h"].min()]
    v = {norm: {kind: int(g["n"].iloc[0]) for kind, g in gg.groupby("kind")}
         for norm, gg in h0.groupby("norm")}
    add("phi.entry.n", "D", "Entry events behind Figure 2, by panel and firm type", v,
        "Number of entry events at h = 0 in each panel of Figure 2 (main: normalized market share; "
        "alt: the alternative normalization) and firm type, as the legend prints them",
        f"DIAG_PHI_SEPARATION/{p.name} n at h = {int(ep['h'].min())} by norm and kind", "ok",
        display="; ".join(f"{n}: " + ", ".join(f"{k} {c}" for k, c in d.items()) for n, d in v.items()))
    drop = {}
    if "main" in v and "alt" in v:
        drop = {k: v["alt"][k] - v["main"][k] for k in v["alt"] if k in v["main"]}
        add("phi.entry.n_dropped", "D", "Entry events the left panel of Figure 2 drops, by firm type",
            drop, "Events in the right panel (s_h/s_end) but not the left: the plateau normalization "
            "(s_h - s_0)/(s_end - s_0) is undefined when s_end is within 1e-6 of s_0",
            f"phi.entry.n.alt - phi.entry.n.main", "ok",
            display=", ".join(f"{k} {c}" for k, c in drop.items()))
    rj = D6_DIR / "d6_drop_robustness.json"
    if not rj.is_file():
        notes.append(f"{rj.name} not found; run sleep_ident_entry_dynamics.py --robustness-only.")
        return
    rb = json.loads(rj.read_text(encoding="utf-8"))
    rv = {k: rb.get(k) for k in ("max_abs_diff", "max_abs_diff_h", "max_abs_width_change",
                                 "max_abs_width_change_h", "n_dropped", "n_censored_both")}
    add("phi.entry.drop_robustness", "D",
        "Figure 2, left panel: the D median with the events the normalization drops forced back in", rv,
        "The D-firm median path under the plateau normalization with and without the events whose "
        "|s_end - s_0| < 1e-6 (included via their actual denominators), same bootstrap seed and B: "
        "the largest absolute change in the median and in the 95% band's width and the quarters h "
        "where they occur; n_dropped: events that guard drops, by firm type; n_censored_both: "
        "events with no quarter in the plateau window, dropped from both panels",
        f"DIAG_PHI_SEPARATION/{rj.name} (sleep_ident_entry_dynamics.py --robustness-only)", "ok",
        display=f"median {fmt3s(rv['max_abs_diff'])} at h={rv['max_abs_diff_h']}; width "
                f"{fmt3s(rv['max_abs_width_change'])} at h={rv['max_abs_width_change_h']}")
    # The robustness run and the published figure must describe the same dropped events.
    if drop and (rb.get("n_dropped") or {}).get("D") != drop.get("D"):
        notes.append(f"{rj.name} drops {(rb.get('n_dropped') or {}).get('D')} D events but the "
                     f"figure drops {drop.get('D')}; re-run sleep_ident_entry_dynamics.py "
                     "--robustness-only.")


# ── output ───────────────────────────────────────────────────────────────────
def _tag_txt(tag) -> str:
    return "(untagged)" if tag == "" else "(none)" if tag is None else tag


def _run_txt(run) -> str:
    if not run:
        return ""
    parts = []
    for r in run:
        b = "--" if r.get("beta") is None else fmt3(r["beta"])
        parts.append(f"{r.get('routine', '')} {_tag_txt(r.get('tag'))} beta {b} T {r.get('T')}"
                     .strip())
    return "; ".join(dict.fromkeys(parts))


def _check_txt(e) -> str:
    if not e["checks"]:
        return "n/a"
    bad = [c for c in e["checks"] if not c["ok"]]
    return "match" if not bad else "MISMATCH"


def write_outputs(meta: dict, notes: list):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for e in ENTRIES:
        e["crosscheck"] = _check_txt(e)
    n_bad = sum(e["crosscheck"] == "MISMATCH" for e in ENTRIES)
    by_status = {}
    for e in ENTRIES:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    doc = dict(meta, entries=ENTRIES, notes=notes, status_summary=by_status,
               crosscheck_summary=dict(
                   match=sum(e["crosscheck"] == "match" for e in ENTRIES),
                   mismatch=n_bad, not_in_a_table=sum(e["crosscheck"] == "n/a" for e in ENTRIES)))
    _dump(OUT_DIR / "paper_numbers.json", json.dumps(_clean(doc), indent=1, ensure_ascii=False) + "\n")

    L = ["# Paper numbers", "",
         f"Generated {meta['generated']} by make_paper_numbers.py. Every value below is read from "
         "a generator's output; none is typed by hand. Values at three decimals (counts as "
         "integers); rate levels in compounded annual percentage points.", "",
         f"- BBL numbers: `{meta['bbl']['file']}` (run tag `{meta['bbl']['tag']}`; "
         f"{meta['bbl']['why']}). Target runs: multi-start `{meta['targets']['multi_start']}`, "
         f"single-curve `{meta['targets']['single_curve']}`.",
         "- Status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())) + ".",
         f"- Cross-check against the rendered tables: {doc['crosscheck_summary']['match']} match, "
         f"{n_bad} MISMATCH, {doc['crosscheck_summary']['not_in_a_table']} not in a table.", ""]
    if notes:
        L += ["**Notes**", ""] + [f"- {n}" for n in notes] + [""]
    L += ["| id | plan | number | value | status | check | run (tag, beta, T) |",
          "|---|---|---|---|---|---|---|"]
    for e in ENTRIES:
        L.append(f"| `{e['id']}` | {e['plan']} | {e['label']} | {e['display'] or '--'} | "
                 f"{e['status']} | {e['crosscheck']} | {_run_txt(e['run']) or '--'} |")
    bad = [(e, c) for e in ENTRIES for c in e["checks"] if not c["ok"]]
    if bad:
        L += ["", "## Cross-check mismatches", ""]
        for e, c in bad:
            L.append(f"- `{e['id']}`: {c['table']} {c['cell']}: expected {c['expected']}, "
                     f"printed {c['printed']} ({c['why']})")
    L += ["", "## Macros", "",
          "`paper_numbers_macros.tex` (PAPER_NUMBERS and Drafts): `\\pn{<id>[.<leaf>]}` prints a "
          "number as the tables do; `\\pn[d]` d = 0-3 decimals, `\\pn[ad]` the absolute value, "
          "`\\pn[pd]` 100 x the value, `\\pn[w]` in words, `\\pn[w10]` rounded to tens in words. "
          "Leaves: `.lo`/`.hi` of an interval, the dotted key of a structured value (e.g. "
          "`bbl.t9.cbar.E3.B.estimate`, `sleep.t4.pix.E3.ci.lo`). Pending numbers print a red ?? "
          "unless `\\PNshowpendingtrue`.", ""]
    L += ["", "## Definitions and sources", ""]
    for e in ENTRIES:
        L.append(f"- `{e['id']}` ({e['plan']}). {e['definition']} Source: {e['source']}."
                 + (f" Unit: {e['unit']}." if e.get("unit") else ""))
    L += ["", "## Refresh after `_ms982` (and `_sc982`) are ingested", "",
          "One line each, from the repo folder:", "",
          "```",
          "python make_bbl_cost_tables.py --psi-tag _ms982 --compare-single-curve "
          "--single-curve-tag _sc982",
          "python make_bbl_cost_tables.py --psi-tag _ms982 --from-psi --psi-zip "
          "<the run's bbl_outputs zip>",
          "julia --project=. bbl_phi_path_summary.jl",
          "python make_paper_numbers.py",
          "```", ""]
    _dump(OUT_DIR / "paper_numbers.md", "\n".join(L))
    return n_bad


# ── LaTeX macros: \pn[<format>]{<id>[.<leaf>]} ───────────────────────────────
_TEX_SPECIALS = str.maketrans({c: "\\" + c for c in "#$%&_{}"} |
                               {"\\": r"\textbackslash{}", "^": r"\^{}", "~": r"\~{}"})


def tex_escape(s: str) -> str:
    return str(s).translate(_TEX_SPECIALS)


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "- - twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int):
    """English words for 0 <= n <= 100 ('forty-three'); None outside that range."""
    if not 0 <= n <= 100:
        return None
    if n == 100:
        return "one hundred"
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (f"-{_ONES[o]}" if o else "")


def _fmt_d(x: float, d: int) -> str:
    """d decimals with thousands separators, never a negative zero."""
    s = f"{x:,.{d}f}"
    return s[1:] if s.startswith("-") and float(s.replace(",", "")) == 0.0 else s


def _tex_num(s: str) -> str:
    """A formatted number for text and math alike: a negative carries a math minus and a
    thousands separator is braced, inside \\ensuremath; anything else stays as it is."""
    neg = s.startswith("-")
    if not neg and "," not in s:
        return s
    body = (s[1:] if neg else s).replace(",", "{,}")
    return r"\ensuremath{" + ("{-}" if neg else "") + body + "}"


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _ambiguous(y: float, known: int, d: int) -> bool:
    """True when a value known only to `known` decimals rounds to `d` decimals differently
    depending on the digits that were rounded away (its discarded part is exactly one half)."""
    if d >= known:
        return False
    f = abs(y) * 10 ** d
    return abs((f - math.floor(f)) - 0.5) < 1e-7


FLOAT_FORMATS = ("0", "1", "2", "3", "a0", "a1", "a2", "a3", "p0", "p1", "p2", "w", "w10")


def tex_formats(x, rounded3=False) -> dict:
    """{format: TeX body} of one leaf; '' is the default (the tables' rule: three decimals, counts
    as integers). A body is None when that format would depend on a digit a 3-dp source rounded
    away. Formats of a number: d = 0-3 decimals; a<d> its absolute value; p<d> 100 x it (a share as
    percent); w |x| rounded to an integer, in words; w10 |x| rounded to tens, in words."""
    if isinstance(x, bool):
        return {"": "yes" if x else "no"}
    if isinstance(x, int):
        out = {"": _tex_num(fmt_count(x))}
        if num_words(abs(x)) is not None:
            out["w"] = num_words(abs(x))
        return out
    if isinstance(x, float):
        out = {"": _tex_num(_fmt_d(x, 3))}
        for f in FLOAT_FORMATS:
            if f in ("w", "w10"):
                tens = f == "w10"
                n = 10 * _half_up(abs(x) / 10) if tens else _half_up(abs(x))
                body = num_words(n)
                amb = rounded3 and (_ambiguous(x / 10, 4, 0) if tens else _ambiguous(x, 3, 0))
            else:
                kind, d = (f[0], int(f[1])) if len(f) == 2 else ("", int(f))
                y = abs(x) if kind == "a" else 100.0 * x if kind == "p" else x
                body = _tex_num(_fmt_d(y, d))
                amb = rounded3 and _ambiguous(y, 1 if kind == "p" else 3, d)
            if body is not None:
                out[f] = None if amb else body
        return out
    return {"": tex_escape(x)}


def _pdf_text(x) -> str:
    """The plain string a PDF bookmark shows for a leaf (the default format, ASCII minus)."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return fmt_count(x)
    if isinstance(x, float):
        return _fmt_d(x, 3)
    return tex_escape(x)


def _py(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _leaves(v, prefix=""):
    """Walk an entry's unrounded value into (dotted.suffix, leaf) pairs. A two-number list is an
    interval (.lo, .hi); a list of strings is one leaf, joined; every entry yields at least one
    leaf (None when it has no value), so every id in paper_numbers.json gets a macro."""
    if isinstance(v, dict):
        got = False
        for k, v2 in v.items():
            for leaf in _leaves(v2, f"{prefix}.{k}" if prefix else str(k)):
                got = True
                yield leaf
        if not got:
            yield prefix, None
        return
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(_is_num(x) for x in v):
            yield (f"{prefix}.lo" if prefix else "lo"), _py(v[0])
            yield (f"{prefix}.hi" if prefix else "hi"), _py(v[1])
            return
        if v and all(isinstance(x, str) for x in v):
            yield prefix, "; ".join(v)
            return
        if not v:
            yield prefix, None
            return
        for i, x in enumerate(v):
            yield from _leaves(x, f"{prefix}.{i}" if prefix else str(i))
        return
    yield prefix, _py(v)


TEX_HEADER = r"""% paper_numbers_macros.tex -- GENERATED by make_paper_numbers.py. Do not edit.
% Generated {generated}. Regenerate: python make_paper_numbers.py
% \input this file in the preamble, after xcolor and hyperref.
%
% \pn{{key}}       the number as the tables print it: 3 decimals, counts as integers, e.g.
%                  \pn{{demand.elast.blp.E3}} -> -1.609. Keys are the ids of paper_numbers.md, plus
%                  .lo/.hi for an interval and the dotted leaf of a structured value.
% \pn[d]{{key}}    d = 0, 1, 2, 3 decimals: \pn[2]{{demand.elast.blp.E3}} -> -1.61
% \pn[ad]{{key}}   the absolute value at d decimals: \pn[a2]{{sleep.t4.selic.E3.estimate}} -> 0.55
% \pn[pd]{{key}}   100 x the value at d decimals, a share as percent: \pn[p0]{{bbl.ud.up.range.lo}}
% \pn[w]{{key}}    |value| rounded to an integer, in words (0-100): \pn[w]{{bbl.t7.blocks.total}} -> four
% \pn[w10]{{key}}  |value| rounded to tens, in words: \pn[w10]{{sleep.pix.awake_rel.E3}} -> forty
% A negative number carries a math minus in text and in math alike. \pn is robust (safe in
% captions); keep it out of section titles (a bookmark shows the 3-decimal value at best).
% A pending, dropped or missing number, an unknown format, and a coarser rounding of a 3-decimal
% table value that would depend on a digit the table rounded away all print a red ?? with a LaTeX
% warning. \PNshowpendingtrue (after \input) shows the provisional value of a pending number in red.
\providecommand{{\PNwarn}}[1]{{\mbox{{\bfseries\color{{red}}??}}\PackageWarning{{paper-numbers}}{{Missing or pending paper number: #1}}}}
\newif\ifPNshowpending
\providecommand{{\PNpending}}[2]{{\ifPNshowpending{{\color{{red}}#2}}\PackageWarning{{paper-numbers}}{{Provisional value shown for the pending paper number #1}}\else\PNwarn{{#1}}\fi}}
\providecommand{{\PNd}}[3]{{\expandafter\def\csname pn@#1@#2\endcsname{{#3}}}}
\DeclareRobustCommand{{\pn}}[2][]{{\ifcsname pn@#2@#1\endcsname\csname pn@#2@#1\endcsname\else\ifcsname pn@#2@\endcsname\PNwarn{{#2 [#1]: no such format for this number}}\else\PNwarn{{#2}}\fi\fi}}
\providecommand{{\PNpdf}}[1]{{\ifcsname pn@#1@pdf\endcsname\csname pn@#1@pdf\endcsname\else ??\fi}}
\def\PNpdfgobble#1#{{\PNpdf}}
\AtBeginDocument{{\ifdefined\pdfstringdefDisableCommands\pdfstringdefDisableCommands{{\let\pn\PNpdfgobble}}\fi}}
"""


def write_tex(meta: dict, dests) -> None:
    """paper_numbers_macros.tex: every leaf of every entry under each format it supports. An `ok`
    leaf expands to its formatted value; a pending one to \\PNpending (?? unless
    \\PNshowpendingtrue); a dropped or absent one to \\PNwarn. Written to each of `dests`."""
    L = [TEX_HEADER.format(generated=meta["generated"])]
    n_ok = n_pend = n_loud = n_amb = 0
    for e in ENTRIES:
        st = e["status"]
        pend = st.startswith("pending")
        for suffix, raw in _leaves(e["value_full"]):
            key = f"{e['id']}.{suffix}" if suffix else e["id"]
            tag = tex_escape(f"{key} ({st})")
            if raw is None or not (st == "ok" or pend):
                L.append(rf"\PNd{{{key}}}{{}}{{\PNwarn{{{tag}}}}}")
                n_loud += 1
                continue
            for f, body in tex_formats(raw, e.get("rounded3", False)).items():
                if body is None:
                    body = (r"\PNwarn{" + tex_escape(f"{key} [{f}]: the table prints 3 decimals and "
                                                     "this rounding depends on the digit it dropped")
                            + "}")
                    n_amb += 1
                elif pend:
                    body = r"\PNpending{" + tag + "}{" + body + "}"
                L.append(rf"\PNd{{{key}}}{{{f}}}{{{body}}}")
            L.append(rf"\PNd{{{key}}}{{pdf}}{{{'??' if pend else _pdf_text(raw)}}}")
            n_pend += pend
            n_ok += not pend
    L += ["", f"% {n_ok} values, {n_pend} pending, {n_loud} missing or dropped, {n_amb} refused "
              f"roundings; {len(ENTRIES)} entries."]
    text = "\n".join(L) + "\n"
    for d in dests:
        _dump(d / "paper_numbers_macros.tex", text)
    print(f"  macros: {n_ok} values, {n_pend} pending, {n_loud} missing/dropped, {n_amb} refused "
          f"roundings (of {len(ENTRIES)} entries)")


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _dump(path: pathlib.Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bbl-tag", default=None,
                    help="read bbl_table_numbers{tag}.json for this psi tag (e.g. _ms1). Default: "
                         "the target run's file when present, else the newest multi-start file.")
    ap.add_argument("--bbl-target", default="_ms982",
                    help="the multi-start run the paper will quote (status ok only for it)")
    ap.add_argument("--tables-dir", default=None,
                    help="where the cross-check reads the rendered tables (default: Drafts)")
    ap.add_argument("--sc-target", default="_sc982",
                    help="the single-curve run Table 11's single-curve columns will come from")
    ap.add_argument("--no-tex", action="store_true",
                    help="skip paper_numbers_macros.tex (written by default to PAPER_NUMBERS and "
                         "to Drafts, next to the tables V_Main inputs)")
    ap.add_argument("--tex", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    global TABLES_DIR
    if a.tables_dir:
        TABLES_DIR = pathlib.Path(a.tables_dir)

    notes: list = []
    path, why = bbl_file(a.bbl_tag, a.bbl_target)
    tag = None
    if path is None:
        notes.append(f"No BBL numbers: {why}. Run make_bbl_cost_tables.py (both passes).")
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))
        tag = collect_bbl(doc, path, a.bbl_target, a.sc_target, notes)
    collect_phi_path(notes)
    collect_demand(notes)
    collect_table_rows(notes)
    phi_base_check(notes)
    collect_entry(notes)
    collect_entry_counts(notes)
    collect_design(notes)
    collect_descriptives(notes)
    collect_polfunc(notes)
    collect_first_stage_strength(notes)
    add("cf.franchise", "E", "Franchise value of deposit funding (R$ x, R$ y, p, q)", None,
        "Removed from the paper by the user on 2026-09-29.", "--", "dropped", display="dropped")
    for t in _TABLES.values():
        if not t.ok:
            notes.append(f"Rendered table {t.name} not found in {TABLES_DIR}; its cross-checks "
                         "fail.")
    meta = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"),
                generator="make_paper_numbers.py",
                targets=dict(multi_start=a.bbl_target, single_curve=a.sc_target),
                bbl=dict(file=str(path) if path else None, tag=tag, why=why,
                         sections=({n: s.get("written") for n, s in
                                    (json.loads(path.read_text(encoding="utf-8")).get("sections")
                                     or {}).items()} if path else None)),
                rendered_tables=str(TABLES_DIR))
    n_bad = write_outputs(meta, notes)
    if not a.no_tex:
        write_tex(meta, [OUT_DIR, DRAFTS])
    for n in notes:
        print(f"  [note] {n}")
    w = max(len(e["id"]) for e in ENTRIES)
    for e in ENTRIES:
        print(f"  {e['id']:<{w}}  {e['status']:<20} {e['crosscheck']:<9} {e['display']}")
    if n_bad:
        print(f"\n  !!!! {n_bad} entr{'y' if n_bad == 1 else 'ies'} MISMATCH a rendered table "
              "(see paper_numbers.md) !!!!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
