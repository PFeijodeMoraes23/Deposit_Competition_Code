#!/usr/bin/env python
r"""
make_bbl_cost_tables.py
=======================
BBL marginal-cost tables, reporting ONLY what the design identifies.

The BBL second stage estimates theta_c = (omega, gamma, zeta) from the moment inequalities in
V_Main eq (17), where psi row 2 loads omega and psi row 4 loads -(1+zeta) on r^f_s * sum(Dep).
Because the BCB Focus forward curve is nearly flat over the DISCOUNTED horizon (beta=0.9 puts
~90% of the weight inside the first ~22 quarters, where rf moves only 0.0333 -> ~0.0285),
d(psi4) and d(psi2) are collinear at corr > 0.9999. omega and zeta are therefore NOT separately
identified; only the combination

    c_bar == omega + rbar_f * zeta        (marginal cost at the mean forward risk-free rate)

is. V_Main line 584 states the failure condition ("a flat rate would leave it unidentified");
this script measures that it binds, and reports c_bar instead of the uninterpretable split.

THE MULTI-START DESIGN CHANGES THAT VERDICT. When psi is simulated from one forward curve per
LAUNCH QUARTER (psi_eq/psi_dev carry a start_q column and the tag ends in _ms<S>), rbar_f differs
across launch quarters, so the pooled d(psi2)/d(psi4) ratio is no longer one number and the ridge
can break. Which regime a run is in is not this script's judgement to make: bbl_solve.py measures
it and records `identified_split`, and the two cost tables -- presentation AND notes -- follow that
flag. The notes below therefore differ between the two branches, because an unqualified "omega and
zeta are not separately identified" becomes false the moment the split is identified.

Writes (to the BBL step folder AND, when it exists, Drafts/Deposit Competition/):
  1. tab_bbl_cost_identified.tex   -- omega/zeta/gamma per routine x firm type. The estimand
                                      table when identified_split is true; a completeness table
                                      reporting a ridge-degenerate split when it is false.
  2. tab_bbl_cbar.tex              -- c_bar and the criterion-health row. c_bar is identified in
                                      every block; it is the only cost object when the split is
                                      not, and restates the split at one rate when it is.
  3. tab_bbl_ridge_diagnostic.tex  -- the collinearity evidence: corr, cond, ratio dispersion,
                                      1-R^2, and the across-Delta spread that rules out a
                                      subsample fix.  (--from-psi)
  4. tab_bbl_ridge_by_start.tex    -- the same design measured LAUNCH QUARTER BY LAUNCH QUARTER,
                                      plus a pooled row: this is where a broken ridge shows up as
                                      the median dpsi4/dpsi2 moving across quarters.  (--from-psi,
                                      multi-start psi only -- it needs the start_q column)
  5. tab_bbl_cbar_design.tex       -- c_bar under the single-curve and the multi-start design.
                                      (--compare-single-curve)
  6. tab_bbl_violated_by_sign.tex  -- APPENDIX: the violated share of tab_bbl_cbar split by the
                                      direction of the deviation (raising vs lowering the firm's
                                      spread).  (--from-psi)

NUMBERS. Every displayed number has exactly three decimals, counts are integers with thousands
separators, -0.000 is never printed, and a nonzero value that prints 0.000 is listed on stdout
(ZERO_PRINTS) rather than rescaled. Rate LEVELS (rbar_f, c_bar, the ratio dpsi4/dpsi2) are shown
as compounded annual percentage points, ((1+x)^4-1)*100; coefficients stay per quarter.

THREE STATISTICS OF ONE RATIO, AND ONLY ONE OF THEM IS rbar_f. Every row of the design is a firm x
launch quarter x signed deviation, and dpsi4/dpsi2 = sum_t beta^t r^f_t dDep_t / sum_t beta^t dDep_t
is the forward rate that row's change in deposits is priced at. The tables summarize it three ways:
  * rbar_f^kappa -- the MEDIAN over one firm type's rows (all quarters), computed by the solve
    (bbl_solve.py rbar_of_block) and the rate c_bar^kappa = omega + rbar_f^kappa * zeta is
    evaluated at. The symbol \bar r^{f,\kappa} is reserved for this object (tab_bbl_cbar,
    tab_bbl_cbar_design).
  * the median over one launch quarter's rows, both types pooled (tab_bbl_ridge_by_start), and
    over every row of a routine (its Pooled row, and the Median column of tab_bbl_ridge_diagnostic).
  * the MEAN and CV over every row of a routine (tab_bbl_ridge_diagnostic), which a handful of
    rows with a nearly cancelling dpsi2 dominate -- reported next to the median and IQR, never
    instead of them.
All of them are taken over the rows with dpsi2 != 0 only (the ratio is undefined elsewhere); when
that leaves the ridge tables' n below the inequality count n_rows of tab_bbl_cost_identified, both
counts are printed in the notes so the two reconcile on the page.

DEAD FIRM-QUARTERS. A (firm, start_q) of a multi-start psi is dead when none of its deviations
moves psi2 beyond round-off (max |dpsi2| <= 1e-12 |psi2_eq|): the firm has no time-deposit or
prepaid product that quarter, so its rows are rounding noise. bbl_solve.py drops them before the
Delta-psi blocks and records the counts; every table follows what the solve did (_null_fq_mode):
the ratio statistics, rbar_f, the violation share and every n are over the remaining rows, and
the notes state the count. A multi-start cost_params written before the rule has no record: the
--from-psi pass then finds the dead firm-quarters in the psi itself, says so, and writes
null_fq_<tag>.json into the step folder, from which the default pass restates the cost tables over
the same rows (_overlay_null_fq). The two passes stay order-free once that file exists.
All are \input-ready in the V_Main house style (spacing + xltabular at \textwidth + booktabs,
caption/label inside the table, notes in \endlastfoot) -- see _wrap() and polfunc_k4.tex.

EVERY TABLE STATES THE beta AND T OF THE RUN IT RENDERS. The Notes end with one sentence built by
_discount_sentence(): the discount factor, its annual equivalent, the forward-simulation horizon
and the share of the discount weight that horizon carries. The values come from the run's own
record (_run_discount: the psi_starts sidecar bbl_fwd_sim.jl writes, tied to the solve by its
per-start rbar_f), never from bbl_discount.env. The registry is read only to append the
derivation of beta when the run used the registry value, and to print a WARNING on stdout when it
did not; a run with no record is stated as such, never filled in.
Also writes markdown twins next to identification_notes.md for the notes document:
  tab_bbl_cost_identified.md, tab_bbl_cbar.md, tab_bbl_ridge_diagnostic.md,
  tab_bbl_ridge_by_start.md, tab_bbl_cbar_design.md, tab_bbl_violated_by_sign.md
and, into the step folder only, bbl_table_numbers{psi_tag}.json: the unrounded numbers each
table prints, from the objects it was rendered from, which make_paper_numbers.py reads (see
_write_numbers). Each pass writes its own sections of that one file.

WHICH PASS PRODUCES THE REPORTED TABLES. Both, and both on the cluster: bbl_job.sh's `tables`
step runs the default pass and then the --from-psi pass against the shards in ${CF_COST_FWD},
and the .tex/.md they write into the BBL step folder are what cluster_archive.sh --set bbl
brings home. The two passes write DISJOINT files, so neither can overwrite the other's.

--from-psi is not a local convenience that could be dropped in favour of reading cost_params:
the ridge tables are per ROUTINE over the pooled design, while cost_params stores its ridge
fields per FIRM TYPE, so the pooled columns (the ratio's median/IQR/mean/CV, #Delta) have no JSON
source -- see the comment at the head of that branch. Run it locally only to inspect a
downloaded psi_cost.zip; the numbers in the paper come from the cluster pass.

Usage:
  python make_bbl_cost_tables.py                       # cost_params only (vintage-safe)
  python make_bbl_cost_tables.py --from-psi            # + the ridge tables, from psi_cost.zip
  python make_bbl_cost_tables.py --from-psi --psi-dir "$CF_COST_FWD"   # on the cluster
  python make_bbl_cost_tables.py --psi-tag _ms8        # pin one multi-start vintage
  python make_bbl_cost_tables.py --psi-tag _ms981 --compare-single-curve --single-curve-tag _sc981
                                                       # tab_bbl_cbar_design's single-curve
                                                       # column from a tagged single-curve solve
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import pathlib
import re
import sys
import zipfile

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import make_blp_rc_table as rc     # est_ref + DRAFTS_DIR + TABLES_DIR
from utils import paths as _paths
from utils import routines as _routines
from utils import bbl_discount as _disc

TABLES_DIR = rc.TABLES_DIR
DRAFTS_DIR = rc.DRAFTS_DIR
EST_OUT = _paths.estimation_output()
PSI_ZIP = _paths.bbl_output_dir() / "cluster_raw" / "psi_cost.zip"
# The BBL step folder: CF_COST_FWD points it at data/output/bbl on the cluster, where
# bbl_solve.py writes cost_params and the archiver packages the step. Locally it is
# ESTIMATION_OUTPUT/COST_FWD, next to the psi the tables describe.
STEP_DIR = _paths.cost_fwd_dir()
# Where cost_params_*.json is READ from. Two real locations, and which one holds the file
# depends on where this runs: on the cluster the solve writes into the step folder, locally
# the downloaded copies land in BBL_OUTPUT/cluster_processed. Resolved at import by
# _resolve_cost_dir() (first candidate that actually holds a matching file), overridable
# with --cost-dir.
COST_DIR = _paths.bbl_output_dir() / "cluster_processed"

ROUTINES = tuple(_routines.LINK_ESTS)
BLOCKS = (("B", r"Brick \& mortar"), ("D", "Digital"))
SPEC, STAGE = _routines.SPEC12_ID, "extended"

# The psi tag that bbl_fwd_sim.jl appends to every artifact of a multi-start run:
# tag = E{estim}_spec_{spec}_{stage}{suffix}{psi_tag}, psi_tag = "_ms{S}" with S rate paths and
# "" without. None means "either", which is what every file matcher below defaults to, so one
# tree holding a single-start and a multi-start vintage is DETECTED rather than half-read.
PSI_TAG = None

# How the solver's `frac_bind` is presented to a reader. Deliberately NOT "share binding":
# in a moment-inequality estimator "binding" means zero slack (g = 0), whereas this counts
# g < 0 -- a strict VIOLATION of the revealed-preference restriction, i.e. a deviation that
# would have raised the firm's value. Calling those "binding" reads as a benign statement
# about which moments are active, when it is the statement that the observed policy is
# dominated. The label works in either regime: near 0 is what optimality implies, 1/2 is the
# mechanical value of a symmetric grid around a non-optimum.
# "violated" rather than "profitable deviations": the latter reads better aloud but is
# positive-valence in a column where a high number is bad, so a skimmer can take 0.47 as "47%
# upside available" — a finding about banks rather than a broken diagnostic. The intuitive gloss
# lives in the note instead. The scriptsize anchor stays true in both regimes: at 0.02 it tells
# the reader the statistic escaped the mechanical value.
FRAC_BIND_LABEL = (r"Share of deviation inequalities violated \newline "
                   r"{\scriptsize ($1/2$ = mechanical)}")
FRAC_BIND_LABEL_MD = "Share of deviation inequalities violated (1/2 = mechanical)"

# The one definition of \bar r^{f,\kappa}, the rate c-bar^kappa = omega + rbar_f * zeta is
# evaluated at. It is bbl_solve.py rbar_of_block: the MEDIAN of dpsi4/dpsi2 over one firm type's
# rows with dpsi2 != 0, all launch quarters and deviations pooled (psi2 = sum_t beta^t Dep_t and
# psi4 = sum_t beta^t r^f_t Dep_t, cf_psi_basis.jl accumulate_psi). The table Notes state it once,
# in tab_bbl_cbar, where the symbol first heads a row, and the other tables refer to that table;
# every other summary of the ratio is named by its statistic ("Median", "Mean") and never by this
# symbol. `rows_note` qualifies the row grain and nothing else. The markdown twins, which are not
# held to the table-note length, keep the full sentence (_rbar_def_md; `dead` adds the row filter
# of a run whose dead firm-quarters are out of the rows, DEAD FIRM-QUARTERS in the docstring).
def _rbar_def(rows_note: str = "", dead_ref: str = "") -> str:
    """The Notes' definition of rbar_f^kappa, stated once (tab_bbl_cbar) and referred to by the
    other tables. `dead_ref` is the clause saying the dead firm-quarters are out of the rows."""
    return (r"$\bar r^{f,\kappa}$ is the median over type-$\kappa$ inequalities" + rows_note
            + (f" ({dead_ref})" if dead_ref else "") + r" of "
            r"$\Delta\psi_4/\Delta\psi_2=\sum_t\beta^t r^f_t\Delta\mathrm{Dep}_t/"
            r"\sum_t\beta^t\Delta\mathrm{Dep}_t$, the forward rate pricing a deviation's deposits")


def _rbar_def_md(rows_note: str = "", dead: bool = False) -> str:
    return ("rbar_f^kappa = the median, over the type-kappa firm × launch-quarter × deviation "
            "rows" + rows_note + " with dpsi_2 != 0"
            + (" outside dead firm-quarters" if dead else "") + ", of dpsi_4/dpsi_2 = "
            "sum_t beta^t r^f_t dDep_t / sum_t beta^t dDep_t, the beta^t dDep_t-weighted mean "
            "forward rate a deviation's change in deposits is priced at")


SINGLE_ROWS = r" (firm~$\times$~deviation in the single-curve design)"
SINGLE_ROWS_MD = " (firm × deviation in the single-curve design)"


def _rbar_label(kappa: str) -> str:
    """Row label for the type-specific rate: $\\bar r^{f,\\mathrm{B}}$ / $\\bar r^{f,\\mathrm{D}}$."""
    return rf"$\bar r^{{f,\mathrm{{{kappa}}}}}$"


# ── file matching ────────────────────────────────────────────────────────────
# Every BBL artifact is named E{estim}_spec_{spec}_{stage}{suffix}{psi_tag}, and the multi-start
# driver is the only thing that fills psi_tag (with _ms{S}, S = rate paths averaged per
# deviation). The matchers below therefore treat the tag as an OPTIONAL trailing group rather
# than as part of a literal filename: a tree can legitimately hold both vintages, and a matcher
# built from an exact name silently reads whichever one it was written for.
def _tag_pat(psi_tag: str | None = None) -> str:
    r"""Regex fragment for the psi tag that sits between the stage and _shard/.parquet/.json.
    None matches either vintage; an explicit tag ("" included) matches only that one."""
    return r"(?:_ms\d+)?" if psi_tag is None else re.escape(psi_tag)


def _tag_of(name: str) -> str:
    """The psi tag actually present in a filename ('' when single-start). Used to REPORT which
    vintage a table was built from, so a mixed tree is visible in the console line."""
    m = re.search(r"_(ms\d+)(?:_shard\d+of\d+)?\.(?:parquet|json)$", name)
    return f"_{m.group(1)}" if m else ""


def _tag_rank(name: str) -> tuple:
    """Sort key that puts the richest vintage first: multi-start before single-start, and the
    largest S first among multi-start runs. Deterministic on purpose -- picking 'whichever the
    glob returned first' is how two vintages of one routine get mixed across tables."""
    t = _tag_of(name)
    return (0 if t else 1, -int(t[3:]) if t else 0, name)


def _psi_names(names, E, psi_tag=None):
    """-> (eq_names, dev_names, cost_names) matching one routine, tag-tolerant."""
    tp = _tag_pat(psi_tag)
    eq = sorted((n for n in names
                 if re.search(rf"psi_eq_E{E}_spec_{SPEC}_{STAGE}{tp}\.parquet$", n)),
                key=_tag_rank)
    dv = sorted(n for n in names
                if re.search(rf"psi_dev_E{E}_spec_{SPEC}_{STAGE}{tp}_shard\d+of\d+\.parquet$", n))
    cp = sorted((n for n in names
                 if re.search(rf"cost_params_E{E}_spec_{SPEC}_{STAGE}{tp}\.json$", n)),
                key=_tag_rank)
    # Only the shards belonging to the chosen eq vintage: psi_eq is written by shard 0 of the
    # SAME run, so its tag is the one the deviation shards have to carry.
    if eq:
        want = _tag_of(eq[0])
        dv = [n for n in dv if _tag_of(n) == want]
    return eq, dv, cp


def _cost_candidates(E, psi_tag=None, cost_dir=None):
    """Every cost_params file for one routine in the resolved cost dir, richest vintage first."""
    d = pathlib.Path(cost_dir or COST_DIR)
    if not d.is_dir():
        return []
    tp = _tag_pat(psi_tag)
    return sorted((p for p in d.glob(f"cost_params_E{E}_spec_{SPEC}_{STAGE}*.json")
                   if re.search(rf"cost_params_E{E}_spec_{SPEC}_{STAGE}{tp}\.json$", p.name)),
                  key=lambda p: _tag_rank(p.name))


def _cost_path(E, psi_tag=None, cost_dir=None):
    """The one cost_params file to read for a routine, or None. Prints when it had a choice:
    a tree holding both a single-start and a multi-start solve of the same routine is a real
    state (a re-run that has not finished every routine yet), and silently taking one of them
    is how half a table ends up describing a different design from the other half."""
    c = _cost_candidates(E, psi_tag, cost_dir)
    if len(c) > 1:
        print(f"  [note] E{E}: {len(c)} cost_params vintages present "
              f"({', '.join(p.name for p in c)}); using {c[0].name}. Pin one with --psi-tag.")
    return c[0] if c else None


def _resolve_cost_dir(psi_tag=None):
    """Which directory actually holds this run's cost_params.

    Two candidates, and the environment says which is real: CF_COST_FWD is exported by
    cl_export_step_dirs on the cluster, where the solve writes into the step folder, so that
    folder is tried first there. Locally the variable is unset and BBL_OUTPUT/cluster_processed
    -- the downloaded copies -- keeps precedence, which is the behaviour every existing table
    was built under. The first candidate holding a matching file wins; when neither does, the
    first is returned so the error message names the place someone should look.
    """
    on_cluster = bool(os.environ.get("CF_COST_FWD", "").strip())
    cands = [STEP_DIR, _paths.bbl_output_dir() / "cluster_processed"]
    if not on_cluster:
        cands.reverse()
    for d in cands:
        if any(_cost_candidates(E, psi_tag, d) for E in ROUTINES):
            return d
    return cands[0]


def _dests():
    """Where a finished table is written: the BBL step folder ALWAYS, Drafts only if it exists.

    The step folder is unconditional because it is the copy that survives: cluster_archive.sh
    packages the step, and on the cluster the Drafts path is the empty skeleton
    cl_bootstrap_tree makes, not the paper. Drafts is conditional because paths.drafts_dir() is
    deliberately never created on demand -- "a copy into a path that does not exist should fail
    rather than scatter fragments into a fresh empty tree" -- so a tree without it must not gain
    one here. Locally both exist and both are written, and the Drafts copy the manuscript
    \\input{}s is unchanged.
    """
    out = [STEP_DIR]
    if DRAFTS_DIR.is_dir():
        out.append(DRAFTS_DIR)
    return out


def _write(name: str, txt: str):
    """Write one table to every destination and say where it went (stdout is the deliverable on
    the cluster, where nothing is downloaded)."""
    for dest in _dests():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(txt, encoding="utf-8")
        print(f"  wrote {name} -> {dest}")


# ── discount factor and horizon of the RENDERED run ─────────────────────────────
# Every table's Notes state the beta and T its psi were simulated under. They are read from the
# run's own record and never from bbl_discount.env: the registry says what the NEXT run uses, and
# a table rendered from an earlier run has to describe that run. Evidence, first hit wins:
#   1. cost_params run.beta / run.T, when the solve records them;
#   2. the psi_starts_<tag>.json sidecar bbl_fwd_sim.jl writes beside psi_eq under --multi-start
#      (beta, T, and the per-start rbar_f it computed from them). A copy counts only when its
#      per-start rbar_f equals the solve's run.rf_bar_beta_by_start exactly -- that ties it to THIS
#      solve rather than to another run that wrote the same tag;
#   3. an archived fwd-sim log that wrote this tag's psi files: its "β=... | T=..." line is parsed
#      and the source printed LOUDLY, because a log is not tied to one solve the way a sidecar is;
#   4. for a run listed in _FIXED_DISCOUNT (matched on its exact rbar_f), the values its code fixed:
#      the 2026-09-01 single-curve solves, which predate every record above;
#   5. nothing -- the Notes say the run recorded no beta/T. A value is never filled in otherwise.
_ZIP_NAMES: dict = {}
_PROV: dict = {}
_BETA_T_LINE = re.compile(r"(?:β|beta)=([0-9.eE+-]+)\s*\|?\s*T=(\d+)")


def _registry():
    """bbl_discount.env as {'beta', 'T', 'rf_mean', 'window'}; None (said once) when unreadable."""
    if "_reg" not in _PROV:
        try:
            _PROV["_reg"] = dict(beta=_disc.beta(), T=_disc.horizon(), rf_mean=_disc.rf_mean_q(),
                                 window=_disc.rf_window())
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [discount] bbl_discount.env unreadable ({type(exc).__name__}: {exc}); the "
                  "Notes state each run's beta/T without the derivation")
            _PROV["_reg"] = None
    return _PROV["_reg"]


def _zip_members(zp: pathlib.Path) -> list:
    """A zip's member names, read once per process (the central directory only, not the data)."""
    k = str(zp)
    if k not in _ZIP_NAMES:
        try:
            with zipfile.ZipFile(zp) as z:
                _ZIP_NAMES[k] = z.namelist()
        except (zipfile.BadZipFile, OSError):
            _ZIP_NAMES[k] = []
    return _ZIP_NAMES[k]


def _sidecar_copies(name: str, extra=()):
    """Yield (where, meta) for every readable copy of one psi_starts sidecar: loose in the cost
    folder, the step folder and any --psi-dir, then inside the --psi-zip and every archive under
    BBL_OUTPUT/cluster_raw."""
    seen = set()
    extra = [pathlib.Path(p) for p in extra]
    for d in [pathlib.Path(COST_DIR), STEP_DIR] + [p for p in extra if p.is_dir()]:
        f = d / name
        if f.is_file() and f.resolve() not in seen:
            seen.add(f.resolve())
            try:
                yield str(f), json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    raw = _paths.bbl_output_dir() / "cluster_raw"
    zips = [p for p in extra if p.is_file()] + (sorted(raw.glob("*.zip")) if raw.is_dir() else [])
    for zp in zips:
        if zp.resolve() in seen:
            continue
        seen.add(zp.resolve())
        for n in _zip_members(zp):
            if n == name or n.endswith("/" + name):
                try:
                    with zipfile.ZipFile(zp) as z:
                        yield f"{zp.name}:{n}", json.loads(z.read(n).decode("utf-8"))
                except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                    continue


def _same_rbar(meta: dict, run: dict):
    """Does a sidecar's per-start rbar_f equal the solve's? None when the solve recorded none."""
    want = {str(k): float(v) for k, v in ((run or {}).get("rf_bar_beta_by_start") or {}).items()}
    if not want:
        return None
    got = {str(q): float(v) for q, v in zip(meta.get("starts") or [], meta.get("rf_bar_beta") or [])}
    return set(got) == set(want) and all(
        abs(got[q] - want[q]) <= 1e-12 * max(1.0, abs(want[q])) for q in want)


def _log_discount(E, tag: str):
    """(beta, T, [logs]) from archived fwd-sim logs that wrote this tag's psi files, or None.
    Searched: CLUSTER_META/*/logs, and the log archives under BBL_OUTPUT/cluster_raw and
    CLUSTER_IN/*. Several logs are expected (one per shard); they must agree."""
    wrote = re.compile(rf"wrote psi_(?:eq|dev)_E{E}_spec_{SPEC}_{STAGE}{re.escape(tag)}"
                       rf"(?:_shard\d+of\d+)?\.parquet")
    fname = re.compile(rf"(?:^|/)bbl_fwd_E{E}_\d+_\d+\.out$")
    found, where = set(), []

    def scan(label, text):
        if wrote.search(text):
            vals = {(float(b), int(t)) for b, t in _BETA_T_LINE.findall(text)}
            if vals:
                found.update(vals)
                where.append(label)

    est = _paths.estimation_output()
    for p in sorted((est / "CLUSTER_META").glob("*/logs/bbl_fwd_E*_*.out")):
        if fname.search(p.as_posix()):
            scan(str(p), p.read_text(encoding="utf-8", errors="replace"))
    zips = (sorted((_paths.bbl_output_dir() / "cluster_raw").glob("*log*.zip"))
            + sorted((est / "CLUSTER_IN").glob("*/*log*.zip"))
            + sorted((est / "CLUSTER_IN").glob("*/_logs_only/*.zip")))
    for zp in zips:
        names = [n for n in _zip_members(zp) if fname.search(n)]
        if names:
            with zipfile.ZipFile(zp) as z:
                for n in names:
                    scan(f"{zp.name}:{n}", z.read(n).decode("utf-8", "replace"))
    if len(found) == 1:
        (b, t), = found
        return b, t, where
    if found:
        print(f"  [discount] E{E}{tag}: archived logs disagree on beta/T {sorted(found)} -- none used")
    return None


# Runs that wrote no beta/T anywhere, with the values the code that produced them fixed, keyed by
# the solve's own theta-free fingerprint (the exact rbar_f of each block) so the entry can only
# ever describe that solve. The single-curve solves of 2026-09-01 (bbl_outputs_24398121.zip,
# cost_params written 15:10): bbl_run.sh then set BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}" and
# bbl_fwd_sim.jl defaulted --beta 0.9 --horizon 50 (commits 0689344b, fa15dc59, ea7a3b2e; the
# registry bbl_discount.env replaced those defaults in 82dc66ec, 2026-09-25). Last tier of
# _run_discount: a record of the run itself always wins.
_FIXED_DISCOUNT_SRC = ("bbl_run.sh / bbl_fwd_sim.jl defaults of the 2026-09-01 single-curve run, "
                       "commit 0689344b")
_FIXED_DISCOUNT = {
    (1, ""): (0.025997269254270875, 0.025648929625908236),
    (2, ""): (0.025992971351086067, 0.02564940437079659),
    (3, ""): (0.025876851026576622, 0.025878098070105132),
    (4, ""): (0.025892517719253515, 0.025860759804732812),
}


def _fixed_discount(E, tag: str, cost: dict):
    """{'beta', 'T', 'source'} for a run in _FIXED_DISCOUNT whose rbar_f match exactly, else None."""
    fp = _FIXED_DISCOUNT.get((int(E), tag or ""))
    got = tuple(((cost or {}).get(k) or {}).get("rbar_f") for k, _ in BLOCKS)
    if fp is None or got != fp:
        return None
    return dict(beta=0.9, T=50, source=_FIXED_DISCOUNT_SRC, fixed=True)


def _cost_tag(cost: dict, default) -> str:
    """The psi tag of the run behind one cost_params: its run record when it has one, else the
    tag the file was pinned with ('' = single curve)."""
    t = ((cost or {}).get("run") or {}).get("psi_tag")
    return t if t is not None else (default or "")


def _run_discount(E, tag: str, cost: dict, extra=()):
    """-> {'beta', 'T', 'source'} for the run behind one routine's cost_params, or None when
    nothing records them (see the block comment above for the order of evidence)."""
    key = (E, tag)
    if key in _PROV:
        return _PROV[key]
    run = (cost or {}).get("run") or {}
    who = f"E{E}{tag}" if tag else f"E{E} (single curve)"
    out = None
    if run.get("beta") is not None and run.get("T") is not None:
        out = dict(beta=float(run["beta"]), T=int(run["T"]), source="cost_params run record")
    if out is None and tag:
        name = f"psi_starts_E{E}_spec_{SPEC}_{STAGE}{tag}.json"
        hits, other = [], 0
        for where, meta in _sidecar_copies(name, extra):
            if meta.get("beta") is None or meta.get("T") is None:
                continue
            same = _same_rbar(meta, run)
            if same is False:
                other += 1
                continue
            hits.append((where, float(meta["beta"]), int(meta["T"]), same))
        vals = {(b, t) for _, b, t, _ in hits}
        if len(vals) == 1:
            where, b, t, same = hits[0]
            how = ("per-start rbar_f equals the solve's" if same else
                   "UNVERIFIED: the solve recorded no per-start rbar_f to match")
            out = dict(beta=b, T=t, source=f"{where} ({how}; {len(hits)} matching, "
                                           f"{other} from other runs skipped)")
        elif vals:
            print(f"  [discount] {who}: matching sidecars disagree {sorted(vals)} -- none used")
    if out is None:
        lg = _log_discount(E, tag)
        if lg:
            b, t, where = lg
            print(f"  !!!! [discount] {who}: beta={b:g} T={t} parsed from {len(where)} archived "
                  f"fwd-sim log(s), first {where[0]} -- a log is not tied to one solve; confirm "
                  "before quoting !!!!")
            out = dict(beta=b, T=t, source=f"archived fwd-sim log {where[0]}")
    if out is None:
        fixed = _fixed_discount(E, tag, cost)
        if fixed:
            print(f"  [discount] {who}: the run recorded no beta/T; its code fixed "
                  f"beta={fixed['beta']:g} T={fixed['T']} ({fixed['source']})")
            out = fixed
    print(f"  [discount] {who}: " + (f"beta={out['beta']:g} T={out['T']} <- {out['source']}"
                                     if out else "NO RECORD of beta/T (no run field, no matching "
                                                 "psi_starts sidecar, no archived fwd-sim log)"))
    _PROV[key] = out
    return out


def _discount_sentence(beta: float, T: int, md: bool = False, lead: str | None = None) -> str:
    """The one sentence every BBL table's Notes use for beta and T. When beta is the registry
    value, its derivation is appended from the registry's BBL_RF_MEAN_Q / BBL_RF_WINDOW."""
    s = "solved at " + _bt(beta, T, md) + "."
    return f"{lead}: {s}" if lead else s[0].upper() + s[1:]


def _bt(beta: float, T: int, md: bool = False) -> str:
    """'$\\beta=0.9$ per quarter ($0.656$ annual) and $T=50$ quarters ($99.5\\%$ of the discount
    weight)', with the registry's derivation of beta when beta is the registry value."""
    reg = _registry()
    b4, cov = _disc.annual(beta), 100.0 * _disc.coverage(beta, T)
    deriv = ""
    if (reg and abs(beta - reg["beta"]) <= 1e-12 and reg.get("rf_mean") is not None
            and reg.get("window")):
        q0, q1 = reg["window"]
        x = 100.0 * reg["rf_mean"]
        deriv = ("; 1/(1 + mean quarterly risk-free rate, " + (f"{x:.3f}%" if md else rf"${x:.3f}\%$")
                 + f", {q0}" + ("-" if md else "--") + f"{q1})")
    # Three decimals on every number (V_Main display rule); the horizon is a count. The coverage is
    # given as a share of the discount weight so that it, too, reads at three decimals.
    if md:
        return (f"beta = {beta:.3f} per quarter ({b4:.3f} annual{deriv}) and T = {T} quarters "
                f"({cov / 100.0:.3f} of the discount weight)")
    return (rf"$\beta={beta:.3f}$ per quarter (${b4:.3f}$ annual{deriv}) and $T={T}$ quarters "
            rf"(${cov / 100.0:.3f}$ of the discount weight)")


def _design_discount_note(Es, prov_single: dict, prov_multi: dict, md: bool = False) -> str:
    """tab_bbl_cbar_design's discount sentence, in the user's wording (2026-09-28): one sentence
    when both designs were solved at the same beta and T, else each design's values and the
    warning that a column difference mixes the design change with the discount change. Falls
    back to _discount_note when the routines of one design differ among themselves."""
    ks, ks_m = ({_prov_key(prov_single.get(E)) for E in Es}, {_prov_key(prov_multi.get(E)) for E in Es})
    if len(ks) != 1 or len(ks_m) != 1:
        return _discount_note(_design_entries(Es, prov_single, prov_multi), md=md)
    ks, km = next(iter(ks)), next(iter(ks_m))
    if ks is not None and ks == km:
        return "Both designs were solved at " + _bt(*ks, md=md) + "."
    if ks is not None and km is not None:
        return ("Single-curve columns were solved at " + _bt(*ks, md=md) + ", multi-start columns "
                "at " + _bt(*km, md=md) + ". The column difference therefore combines the design "
                "change and the discount change.")
    have, lab_have, lab_none = ((ks, "Single-curve", "multi-start") if ks is not None else
                                (km, "Multi-start", "single-curve"))
    if have is None:
        return ("No beta or T is recorded for either design." if md else
                r"No $\beta$ or $T$ is recorded for either design.")
    return (f"{lab_have} columns were solved at " + _bt(*have, md=md) + "; "
            + ("no beta or T is recorded for the " if md else r"no $\beta$ or $T$ is recorded for the ")
            + f"{lab_none} run.")


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _routine_entries(Es, prov: dict) -> list:
    """One footnote entry per routine: (tex lead, md lead), (tex noun, md noun), provenance."""
    return [dict(lead=(rc.est_ref(E), f"E{E}"), noun=(rf"the {rc.est_ref(E)} run", f"the E{E} run"),
                 prov=prov.get(E)) for E in Es]


def _design_entries(Es, prov_single: dict, prov_multi: dict) -> list:
    """tab_bbl_cbar_design: one entry per design, or per design x routine when a design's
    routines were simulated under different beta/T."""
    out = []
    for lead, noun, prov in (((r"\emph{Single curve}", "Single curve"), "single-curve", prov_single),
                             ((r"\emph{Multi-start}", "Multi-start"), "multi-start", prov_multi)):
        ps = [prov.get(E) for E in Es]
        if len({_prov_key(p) for p in ps}) <= 1:
            out.append(dict(lead=lead, noun=(f"the {noun} run", f"the {noun} run"),
                            prov=ps[0] if ps else None))
            continue
        for E, p in zip(Es, ps):
            out.append(dict(lead=(rf"{lead[0]} {rc.est_ref(E)}", f"{lead[1]} E{E}"),
                            noun=(rf"the {noun} {rc.est_ref(E)} run", f"the {noun} E{E} run"),
                            prov=p))
    return out


def _prov_key(p):
    return None if p is None else (float(p["beta"]), int(p["T"]))


def _discount_note(entries: list, md: bool = False, all_lead=None) -> str:
    """Footnote text: one sentence when every entry was simulated under the same recorded beta/T,
    else one sentence per distinct value (led by the entries that share it) plus a statement of
    the entries with no record."""
    i = 1 if md else 0
    keys = [_prov_key(e["prov"]) for e in entries]
    if keys and None not in keys and len(set(keys)) == 1:
        return _discount_sentence(*keys[0], md=md, lead=all_lead[i] if all_lead else None)
    parts = [_discount_sentence(*k, md=md,
                                lead=_join(e["lead"][i] for e, kk in zip(entries, keys) if kk == k))
             for k in dict.fromkeys(k for k in keys if k is not None)]
    missing = [e["noun"][i] for e, k in zip(entries, keys) if k is None]
    if missing:
        parts.append(("No beta or T is recorded for " if md else r"No $\beta$ or $T$ is recorded for ")
                     + _join(missing) + ".")
    return " ".join(parts)


def _discount_report(table: str, entries: list, all_lead=None) -> tuple:
    """-> (tex, md) footnote sentences for one table, after printing a WARNING for every entry
    whose run used a beta/T other than the registry's, or recorded none."""
    reg = _registry()
    for e in entries:
        p, who = e["prov"], f"{table} [{e['lead'][1]}]"
        if p is None:
            print(f"  WARNING: {who}: no beta/T recorded for this run; the Notes say so and no "
                  "value is filled in.")
        elif reg and (abs(p["beta"] - reg["beta"]) > 1e-12 or p["T"] != reg["T"]):
            print(f"  WARNING: {who}: the rendered run was simulated at beta={p['beta']:g}, "
                  f"T={p['T']}; bbl_discount.env sets BBL_BETA={reg['beta']:g}, "
                  f"BBL_HORIZON={reg['T']}. The Notes state the run's values.")
    return (_discount_note(entries, md=False, all_lead=all_lead),
            _discount_note(entries, md=True, all_lead=all_lead))


def _with(note: str, disc: str) -> str:
    """Append the discount sentence to a table's Notes."""
    return note + (" " + disc if disc else "")


# bbl_solve.py's two resampling schemes behind the SDs, intervals and stars of Tables 7, 9 and 11.
# The draw counts are recorded by the solve (run.bootstrap; each block's subsample.n_sub and
# b_firms). The seeds are neither options nor recorded: bootstrap_kappa and subsample_kappa run
# with their signature defaults, 42 and 4242, the same in every commit since 0689344b
# (2026-08-29), so they are restated here and must follow bbl_solve.py if its defaults change.
SOLVE_BOOT_SEED = 42
SOLVE_SUB_SEED = 4242


def _resampling_parts(costs, md: bool = False) -> dict:
    """{'boot': '200 draws, seed 42', 'sub': '200 draws of $b=n^{2/3}$ firms, seed 4242'} from the
    solves' own records; a count a solve did not record is said to be so, and a scheme no solve
    ran has no key. Table 7 embeds the two clauses where it names the intervals and the SDs;
    _resampling_note makes the sentence the other tables carry."""
    costs = [c for c in costs if c]
    nb = {(c.get("run") or {}).get("bootstrap") for c in costs}
    subs = [((c.get(k) or {}).get("subsample") or {}) for c in costs for k in ("B", "D")]
    subs = [s for s in subs if s.get("n_sub")]
    ns = sorted({int(s["n_sub"]) for s in subs})
    rule = all(int(s["b_firms"]) == int(np.clip(round(s["n_firms"] ** (2.0 / 3.0)), 5,
                                                 s["n_firms"] - 1))
               for s in subs if s.get("n_firms") and s.get("b_firms"))
    parts = {}
    if nb - {None, 0}:
        cnt = "/".join(f"{int(x):,}" for x in sorted(nb - {None, 0}))
        parts["boot"] = (f"{cnt} draws" + (" (not recorded for every solve)" if None in nb else "")
                         + f", seed {SOLVE_BOOT_SEED}")
    elif costs and nb == {None}:
        parts["boot"] = f"draw count not recorded, seed {SOLVE_BOOT_SEED}"
    if ns:
        b = ("b = n^(2/3)" if md else r"$b=n^{2/3}$") if rule else "b"
        parts["sub"] = (f"{'/'.join(f'{x:,}' for x in ns)} draws of {b} firms, "
                        f"seed {SOLVE_SUB_SEED}")
    return parts


def _resampling_note(costs, md: bool = False) -> str:
    """'Firm-block bootstrap: 200 draws, seed 42; subsampling: 200 draws of b = n^(2/3) firms,
    seed 4242.' from _resampling_parts; empty when neither scheme ran."""
    p = _resampling_parts(costs, md=md)
    parts = ([f"firm-block bootstrap: {p['boot']}"] if "boot" in p else []) \
        + ([f"subsampling: {p['sub']}"] if "sub" in p else [])
    if not parts:
        return ""
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


def _notes_join(*parts) -> str:
    """Sentences joined by one space, empty ones skipped."""
    return " ".join(p for p in parts if p)


# ── data ─────────────────────────────────────────────────────────────────────
def load_from_zip(zpath: pathlib.Path, psi_tag: str | None = None):
    """-> {routine: {'psi_eq': df, 'psi_dev': df, 'cost': dict}}. Reads the archive in place so
    400 shard parquets are not duplicated onto disk."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()

        def _read_parquet(n):
            return pd.read_parquet(io.BytesIO(z.read(n)))

        for E in ROUTINES:
            eq_n, dv_n, cp_n = _psi_names(names, E, psi_tag)
            # cost_params come from the resolved cost dir, NOT the archive: the archived copies
            # predate the c_bar/ridge fields that bbl_solve.py now persists.
            # Fall back to the archived copy so the script still runs on an untouched tree.
            cp_live = _cost_path(E, psi_tag)
            if not (eq_n and dv_n and (cp_live or cp_n)):
                print(f"  [skip] E{E}: eq={len(eq_n)} dev={len(dv_n)} "
                      f"cost={bool(cp_live) or len(cp_n)}")
                continue
            cost = (json.loads(cp_live.read_text(encoding="utf-8")) if cp_live
                    else json.loads(z.read(cp_n[0]).decode("utf-8")))
            out[E] = {
                "psi_eq": _read_parquet(eq_n[0]),
                "psi_dev": pd.concat([_read_parquet(n) for n in dv_n], ignore_index=True),
                "cost": cost,
                "n_shards": len(dv_n),
                "psi_tag": _tag_of(eq_n[0]),
            }
    return out


def load_from_dir(dpath: pathlib.Path, psi_tag: str | None = None):
    """The same load, from LOOSE psi parquets in a directory.

    This is the cluster path. Nothing is downloaded from Bouchet, so the tables are built where
    the shards are written -- data/output/bbl, before cluster_archive.sh has packaged anything --
    and a loader that only understands psi_cost.zip would make the ridge tables producible on the
    local machine alone, which is the one machine that will never see the psi again.
    """
    out = {}
    names = sorted(p.name for p in dpath.glob("psi_*.parquet"))
    for E in ROUTINES:
        eq_n, dv_n, _ = _psi_names(names, E, psi_tag)
        cp_live = _cost_path(E, psi_tag)
        if not (eq_n and dv_n and cp_live):
            print(f"  [skip] E{E}: eq={len(eq_n)} dev={len(dv_n)} cost={bool(cp_live)}")
            continue
        out[E] = {
            "psi_eq": pd.read_parquet(dpath / eq_n[0]),
            "psi_dev": pd.concat([pd.read_parquet(dpath / n) for n in dv_n], ignore_index=True),
            "cost": json.loads(cp_live.read_text(encoding="utf-8")),
            "n_shards": len(dv_n),
            "psi_tag": _tag_of(eq_n[0]),
        }
    return out


def load_psi(path: pathlib.Path, psi_tag: str | None = None):
    """Dispatch on what `path` IS: a directory of loose shards (cluster) or psi_cost.zip (local)."""
    return (load_from_dir(path, psi_tag) if path.is_dir()
            else load_from_zip(path, psi_tag))


def _merge_keys(eq: pd.DataFrame, dev: pd.DataFrame) -> list:
    r"""The columns that identify one psi row on both sides of the equilibrium/deviation join.

    (firm, is_B) alone is right only for a single-start run. Under multi-start, accumulate_psi
    groups psi by (firm, start) -- one row per firm per LAUNCH QUARTER -- so joining on the firm
    alone forms a 36x36 cross product of quarters that never shared a share denominator, and
    every column below would then be measured on rows the design never paired. start_q is added
    to the key whenever both frames carry it.
    """
    keys = ["firm", "is_B"]
    if "start_q" in eq.columns and "start_q" in dev.columns:
        keys.append("start_q")
    return keys


# ── dead firm-quarters ───────────────────────────────────────────────────────
# The rule is bbl_solve.py's (dead_fq_mask / drop_dead_fq), restated here rather than imported:
# importing bbl_solve runs its venv guard, which re-executes that FILE when the interpreter is not
# the project venv. test_null_fq_mask.py checks the two masks agree row for row.
#   solve_drop  the solve dropped them (run.null_fq.path multi_start_drop): cost_params is already
#               over the live rows; --from-psi applies the mask with the recorded tol and checks
#               the counts against the solve's.
#   solve_keep  --keep-null-fq: the solve kept them, and so do the tables.
#   table_mask  a multi-start cost_params written before the rule (no run.null_fq): --from-psi
#               finds them in the psi, and the default pass restates the cost tables from the
#               record that pass writes (_null_fq_record / _overlay_null_fq).
#   none        a single-curve psi, which has no start_q of its own: the rule does not apply.
DEAD_FQ_TOL = 1e-12          # bbl_solve.DEAD_FQ_TOL: |dpsi2| <= tol * |psi2_eq| on every deviation
# The Notes' definition: the table notes are held to a few lines (V_Main style guide, 2026-09-28),
# so the tolerance lives in the .md twins and the code, and the notes give the economic content.
DEAD_FQ_DEF = r"firm and launch-quarter pairs with no deposits of the deviated type"
DEAD_FQ_DEF_MD = ("a firm × launch quarter is dead when none of its deviations moves psi_2 by more "
                  "than 1e-12 |psi_2|, i.e. the firm has no time-deposit (k=4) or prepaid (k=5) "
                  "product that quarter, so no spread deviation reaches its deposits and its "
                  "inequalities are rounding noise")


def _null_fq_mode(cost: dict, multi: bool | None = None) -> str:
    """Which of the four paths above a routine is on. `multi` is whether its psi carries its own
    launch quarters; None infers it from the cost_params run record (the default pass has no psi)."""
    run = (cost or {}).get("run") or {}
    rec = run.get("null_fq")
    if isinstance(rec, dict):
        path = rec.get("path")
        if path == "table_mask":                 # a cost_params restated by _overlay_null_fq
            return "table_mask"
        if path == "multi_start_keep" or run.get("keep_null_fq"):
            return "solve_keep"
        if path == "multi_start_drop" or (path is None and rec.get("applied")):
            return "solve_drop"
        return "none"
    if multi is None:
        multi = str(run.get("psi_tag") or "").startswith("_ms") or bool(run.get("n_starts"))
    return "table_mask" if multi else "none"


def _null_fq_tol(cost: dict) -> float:
    """The tolerance the solve applied, when it recorded one; the solve's constant otherwise."""
    t = (((cost or {}).get("run") or {}).get("null_fq") or {}).get("tol")
    return float(t) if t is not None else DEAD_FQ_TOL


def dead_fq_mask(eq: pd.DataFrame, dev: pd.DataFrame, tol: float = DEAD_FQ_TOL) -> np.ndarray:
    """Boolean over the rows of dev: True where the row's (firm, start_q) is dead, i.e. every one
    of its deviations leaves psi2 within tol*|psi2_eq| of equilibrium (bbl_solve.dead_fq_mask).
    A dev row with no psi_eq partner, or a non-finite psi2, fails the comparison and so is never
    dead, as in the solve."""
    sq = dev["start_q"].astype(str)
    e = eq.assign(start_q=eq["start_q"].astype(str)).set_index(["firm", "start_q"])["psi2_omega"]
    if e.index.has_duplicates:
        raise ValueError("psi_eq has duplicate (firm, start_q) rows")
    p2e = e.reindex(pd.MultiIndex.from_arrays([dev["firm"].to_numpy(), sq.to_numpy()])).to_numpy(float)
    quiet = np.abs(p2e - dev["psi2_omega"].to_numpy(float)) <= tol * np.abs(p2e)
    return (pd.Series(quiet, index=dev.index)
            .groupby([dev["firm"], sq], sort=False, dropna=False)
            .transform("all").to_numpy(bool))


def _dead_counts(dev: pd.DataFrame, dead: np.ndarray) -> dict:
    """{kappa: {n_null_fq, n_null_rows}} over the firm types present in dev, counted as the solve
    counts them (a firm-quarter under the is_B of its rows)."""
    isb = dev["is_B"].to_numpy().astype(bool)
    out = {}
    for kappa, mk in (("B", isb), ("D", ~isb)):
        if not mk.any():
            continue
        dm = dead & mk
        out[kappa] = dict(n_null_fq=int(len(dev.loc[dm, ["firm", "start_q"]].drop_duplicates())),
                          n_null_rows=int(dm.sum()))
    return out


def _solve_dead_counts(cost: dict) -> dict:
    """The per-type counts the solve recorded: the block fields, else run.null_fq.by_type."""
    by = ((((cost or {}).get("run") or {}).get("null_fq") or {}).get("by_type")) or {}
    out = {}
    for kappa, _lbl in BLOCKS:
        blk, alt = (cost or {}).get(kappa) or {}, by.get(kappa) or {}
        fq, rows = blk.get("n_null_fq", alt.get("n_null_fq")), blk.get("n_null_rows",
                                                                     alt.get("n_null_rows"))
        if fq is not None or rows is not None:
            out[kappa] = dict(n_null_fq=fq, n_null_rows=rows)
    return out


def _g_rows(m: pd.DataFrame, blk: dict) -> np.ndarray:
    """g = dpsi1 - omega dpsi2 - gamma'dpsi3 - (1+zeta) dpsi4 at one block's theta-hat, over the
    rows of a psi_dev x psi_eq merge (suffixes _d/_e; Delta-psi = psi_eq - psi_dev, as in
    bbl_solve.build_delta). Its sign is the solve's frac_bind test."""
    d = lambda c: (m[c + "_e"] - m[c + "_d"]).to_numpy(float)  # noqa: E731
    g = d("psi1") - float(blk["omega"]) * d("psi2_omega") - (1.0 + float(blk["zeta"])) * d("psi4_zeta")
    for c, v in (blk.get("gamma") or {}).items():
        if c + "_e" in m.columns:
            g = g - float(v) * d(c)
    return g


def _cbar_se_at(blk: dict, rbar: float):
    """The firm-block bootstrap SD of omega + rbar*zeta from the recorded SDs and correlation.

    Exact rather than approximate: the solve's c_bar_se_boot is std(omega_b + rbar*zeta_b) over the
    draws and omega_se/zeta_se/omega_zeta_corr_boot are the same draws' std and correlation, so
    the SD at any rate is the quadratic form. Returned only when it reproduces the solve's own
    figure at the solve's rate; None otherwise."""
    so, sz, rho = blk.get("omega_se"), blk.get("zeta_se"), blk.get("omega_zeta_corr_boot")
    if None in (so, sz, rho, blk.get("c_bar_se_boot"), blk.get("rbar_f")):
        return None

    def sd(r):
        return float(np.sqrt(max(so * so + r * r * sz * sz + 2.0 * r * rho * so * sz, 0.0)))
    base = float(blk["c_bar_se_boot"])
    if abs(sd(float(blk["rbar_f"])) - base) > 1e-9 * max(1.0, abs(base)):
        return None
    return sd(float(rbar))


def _merged(eq: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    return dev.merge(eq, on=_merge_keys(eq, dev), suffixes=("_d", "_e"))


def _null_fq_record(E, eq: pd.DataFrame, dev: pd.DataFrame, dead: np.ndarray, cost: dict,
                    source: str) -> dict:
    """What the default pass needs to restate one routine's cost tables without the dead
    firm-quarters, for a cost_params that predates the rule (table_mask). Computed from the psi at
    the solve's own theta-hat; `solve` fingerprints the cost_params it belongs to, and every
    all-row figure is recomputed here first, so a psi that is not the solve's is refused."""
    m = _merged(eq, dev.assign(_dead=dead))
    isb = m["is_B"].to_numpy().astype(bool)
    dead_m = m["_dead"].to_numpy(bool)
    d2 = (m["psi2_omega_e"] - m["psi2_omega_d"]).to_numpy(float)
    d4 = (m["psi4_zeta_e"] - m["psi4_zeta_d"]).to_numpy(float)
    okr = np.isfinite(d2) & np.isfinite(d4) & (d2 != 0)
    ratio = np.where(okr, d4 / np.where(okr, d2, 1.0), np.nan)
    sq = m["start_q"].astype(str).to_numpy()
    rec = dict(kind="null_fq_overlay", E=int(E), tag=_cost_tag(cost, PSI_TAG), psi_source=source,
               rule=("max over a firm-quarter's deviations of |psi2_eq - psi2_dev| <= "
                     "tol * |psi2_eq|"), tol=DEAD_FQ_TOL, by_type={}, solve={})
    for kappa, mk in (("B", isb), ("D", ~isb)):
        blk = (cost or {}).get(kappa)
        if not blk or not mk.any():
            continue
        live = mk & ~dead_m
        g = _g_rows(m.loc[mk], blk)
        fb_all = float(np.mean(g < 0))
        if abs(fb_all - float(blk["frac_bind"])) > 1e-9:
            raise ValueError(f"E{E}-{kappa}: frac_bind at the solve's theta over all rows is "
                             f"{fb_all:.10f} here, {blk['frac_bind']:.10f} in cost_params")
        st = ridge_stats(None, None, m=m.loc[live])
        per = {}
        for q in sorted(set(sq[mk])):
            inq = mk & (sq == q)
            lq = inq & ~dead_m
            rq = ratio[lq & okr]
            per[q] = dict(n=int(lq.sum()), n_ratio=int(rq.size), n_null_rows=int((inq & dead_m).sum()),
                          rbar_f=float(np.median(rq)) if rq.size else None,
                          all_dead=bool(inq.any() and not lq.any()))
        rec["solve"][kappa] = {k: blk.get(k) for k in ("n_rows", "n_firms", "rbar_f", "frac_bind",
                                                      "omega", "zeta", "c_bar", "c_bar_se_boot")}
        rec["by_type"][kappa] = dict(
            **_dead_counts(m.loc[mk], dead_m[mk])[kappa],
            n_rows=int(live.sum()), n_ratio=int((live & okr).sum()),
            n_firms=int(m.loc[live, "firm"].nunique()),
            rbar_f=float(np.median(ratio[live & okr])),
            frac_bind=float(np.mean(g[~dead_m[mk]] < 0)),
            pooled_corr=st["corr"], pooled_bkw_cond=st["cond"],
            g_dead_absmax=float(np.max(np.abs(g[dead_m[mk]]))) if dead_m[mk].any() else 0.0,
            per_start=per)
    return rec


def _null_fq_record_path(E, tag: str) -> pathlib.Path:
    return STEP_DIR / f"null_fq_E{E}_spec_{SPEC}_{STAGE}{tag}.json"


def _read_null_fq_record(E, cost: dict):
    """The table_mask record for one routine, when it exists AND fingerprints this cost_params;
    None (said on stdout) otherwise."""
    p = _null_fq_record_path(E, _cost_tag(cost, PSI_TAG))
    if not p.is_file():
        print(f"  !!!! [dead fq] E{E}: cost_params predates the dead firm-quarter rule and "
              f"{p.name} is missing -- the cost tables keep the dead firm-quarters. Run the "
              f"--from-psi pass (it writes that file), then this pass again. !!!!")
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    for kappa, fp in (rec.get("solve") or {}).items():
        blk = (cost or {}).get(kappa) or {}
        bad = [k for k, v in fp.items() if blk.get(k) != v]
        if bad:
            print(f"  !!!! [dead fq] E{E}-{kappa}: {p.name} was computed for another solve "
                  f"({', '.join(bad)} differ) -- ignored; the cost tables keep the dead "
                  f"firm-quarters. !!!!")
            return None
    print(f"  [dead fq] E{E}: cost_params predates the rule; restating it without the dead "
          f"firm-quarters from {p.name} (computed from {rec.get('psi_source')})")
    return rec


def _overlay_null_fq(cost: dict, rec: dict) -> dict:
    """A copy of one routine's cost_params restated over the live rows (table_mask).

    theta-hat and its intervals are the solve's: the dead rows' g is rounding noise, so they do
    not load the criterion. What changes is every figure that counts or averages rows: n_rows,
    frac_bind, rbar_f, the pooled corr/condition index, and c-bar evaluated at the new rbar_f,
    with its bootstrap SD re-evaluated exactly (_cbar_se_at). The subsampling interval of c-bar
    cannot be: its draws are not stored, so it stays the solve's, evaluated at the solve's rbar_f,
    which `_ci_rbar` records for the Notes. Per launch quarter the refit's (omega, zeta) are kept
    and c-bar re-evaluated at the quarter's live median; its frac_bind, which would need the
    refit's gamma, is dropped."""
    out = json.loads(json.dumps(cost))
    for kappa, t in (rec.get("by_type") or {}).items():
        blk = out.get(kappa)
        if not blk:
            continue
        w, z, r_new = float(blk["omega"]), float(blk["zeta"]), float(t["rbar_f"])
        se = _cbar_se_at(blk, r_new)
        blk["_ci_rbar"] = blk.get("rbar_f")
        blk.update(n_rows=t["n_rows"], n_firms=t["n_firms"], n_null_fq=t["n_null_fq"],
                   n_null_rows=t["n_null_rows"], rbar_f=r_new, frac_bind=t["frac_bind"],
                   c_bar=w + r_new * z)
        if se is not None:
            blk["c_bar_se_boot"] = se
        else:
            print(f"  !!!! [dead fq] {kappa}: the bootstrap SD of c-bar does not reproduce from "
                  f"omega_se/zeta_se/corr -- left at the solve's rate !!!!")
        bs = blk.get("by_start")
        if isinstance(bs, dict):
            bs["pooled_corr"], bs["pooled_bkw_cond"] = t["pooled_corr"], t["pooled_bkw_cond"]
            pooled = bs.get("pooled")
            if isinstance(pooled, dict):
                pooled.update(n=t["n_rows"], n_ratio=t["n_ratio"], rbar_f=r_new,
                              corr=t["pooled_corr"], bkw_cond=t["pooled_bkw_cond"],
                              frac_bind=t["frac_bind"], c_bar=w + r_new * z)
            per = bs.get("per_start") if isinstance(bs.get("per_start"), dict) else None
            for q, pq in (t.get("per_start") or {}).items():
                src = (per or {}).get(q)
                if per is None or src is None:
                    continue
                if pq["all_dead"]:
                    per[q] = dict(n=0, n_ratio=0, all_dead=True)
                    continue
                cb = (float(src["omega"]) + pq["rbar_f"] * float(src["zeta"])
                      if pq["rbar_f"] is not None and src.get("omega") is not None else None)
                per[q] = dict(src, n=pq["n"], n_ratio=pq["n_ratio"], rbar_f=pq["rbar_f"],
                              c_bar=cb, frac_bind=None, all_dead=False)
        ident = (blk.get("by_start") or {}).get("identified_split", blk.get("identified_split"))
        thr = float((blk.get("by_start") or {}).get("ridge_cond_max") or 30.0)
        if ident is not None and bool(ident) != (t["pooled_bkw_cond"] <= thr):
            print(f"  !!!! [dead fq] {kappa}: without the dead firm-quarters the pooled condition "
                  f"index is {t['pooled_bkw_cond']:.1f}, the other side of {thr:g} from the "
                  f"solve's verdict -- the verdict shown is the solve's !!!!")
    run = out.setdefault("run", {})
    run["null_fq"] = dict(path="table_mask", applied=True, tol=rec.get("tol"), rule=rec.get("rule"),
                          by_type={k: dict(n_null_fq=v["n_null_fq"], n_null_rows=v["n_null_rows"])
                                   for k, v in (rec.get("by_type") or {}).items()},
                          source=rec.get("psi_source"))
    return out


def apply_null_fq(data: dict) -> None:
    """The --from-psi side of the rule, routine by routine, in place. Adds to each entry:
      psi_dev_solve  the rows the solve used (what check_psi_matches_solve compares against),
      psi_dev        the rows the tables describe,
      psi_dev_all    every row loaded,
      cost_solve     the cost_params as the solve wrote it,
      null           {mode, tol, dropped, n_loaded, dead_rows, dead_fq, counts, solve_counts,
                      dead_q}: dead_q is {quarter: [firm types all of whose firm-quarters are
                      dead there]}, the rows of tab_bbl_ridge_by_start dashed for that reason.
    A table_mask routine keeps its every-row frame as psi_dev until restate_table_mask(), which
    runs after the vintage check, since its record is computed at the solve's own theta-hat."""
    for E, d in sorted(data.items()):
        eq, dev, cost = d["psi_eq"], d["psi_dev"], d["cost"]
        multi = "start_q" in eq.columns and "start_q" in dev.columns
        mode = _null_fq_mode(cost, multi=multi)
        if not multi and mode != "none":
            print(f"  !!!! [dead fq] E{E}: cost_params records the rule ({mode}) but the psi has no "
                  f"start_q of its own -- no mask applied !!!!")
            mode = "none"
        d.update(cost_solve=cost, psi_dev_solve=dev, psi_dev_all=dev)
        nf = dict(mode=mode, n_loaded=int(len(dev)), dropped=False, dead_q={})
        d["null"] = nf
        if mode == "none":
            continue
        tol = _null_fq_tol(cost)
        dead = dead_fq_mask(eq, dev, tol)
        counts = _dead_counts(dev, dead)
        g = (pd.DataFrame({"q": dev["start_q"].astype(str).to_numpy(),
                           "t": np.where(dev["is_B"].to_numpy().astype(bool), "B", "D"),
                           "dead": dead})
             .groupby(["q", "t"])["dead"].all())
        dead_q = {}
        for (q, t), v in g.items():
            if v:
                dead_q.setdefault(q, []).append(t)
        nf.update(tol=tol, counts=counts, dead_q=dead_q, dead_mask=dead,
                  dead_rows=int(dead.sum()),
                  dead_fq=int(sum(c["n_null_fq"] for c in counts.values())))
        lst = ", ".join(f"{k} {c['n_null_fq']:,} fq / {c['n_null_rows']:,} rows"
                        for k, c in counts.items())
        if mode == "solve_drop":
            nf["solve_counts"] = _solve_dead_counts(cost)
            live = dev.loc[~dead].reset_index(drop=True)
            d.update(psi_dev_solve=live, psi_dev=live)
            nf["dropped"] = True
            print(f"  [dead fq] E{E}: the solve dropped the dead firm-quarters (tol {tol:g}); the "
                  f"psi mask finds {lst} -- checked against the solve's counts below")
        elif mode == "solve_keep":
            print(f"  [dead fq] E{E}: the solve KEPT the dead firm-quarters (--keep-null-fq), and "
                  f"so do the tables; the psi mask finds {lst}")
        else:
            print(f"  [dead fq] E{E}: cost_params predates the dead firm-quarter rule (no "
                  f"run.null_fq), so the mask is computed here from the psi (tol {tol:g}): {lst}")


def restate_table_mask(data: dict, source: str) -> None:
    """For every table_mask routine: build the record at the solve's theta-hat, restate the
    cost_params from it for the tables of this pass, and write it where the default pass reads it.
    The file is rewritten only when its content changes, and a change is announced, because the
    default pass has to run again to carry it into tab_bbl_cost_identified / _cbar / _cbar_design."""
    for E, d in sorted(data.items()):
        nf = d.get("null") or {}
        if nf.get("mode") != "table_mask":
            continue
        dead = nf["dead_mask"]
        rec = _null_fq_record(E, d["psi_eq"], d["psi_dev_all"], dead, d["cost_solve"], source)
        d["cost"] = _overlay_null_fq(d["cost_solve"], rec)
        d["psi_dev"] = d["psi_dev_all"].loc[~dead].reset_index(drop=True)
        nf["dropped"] = True
        for kappa, t in rec["by_type"].items():
            s = rec["solve"][kappa]
            print(f"  [dead fq] E{E}-{kappa}: n {s['n_rows']:,} -> {t['n_rows']:,}; frac_bind "
                  f"{s['frac_bind']:.4f} -> {t['frac_bind']:.4f}; rbar_f {s['rbar_f']:.5f} -> "
                  f"{t['rbar_f']:.5f}; pooled cond {t['pooled_bkw_cond']:.2f}; max |g| over dead "
                  f"rows {t['g_dead_absmax']:.2e}")
        p = _null_fq_record_path(E, rec["tag"])
        txt = json.dumps(rec, indent=1, sort_keys=True)
        old = p.read_text(encoding="utf-8") if p.is_file() else None
        if old == txt:
            print(f"  [dead fq] E{E}: {p.name} unchanged")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
        print(f"  !!!! [dead fq] E{E}: wrote {p} -- run the default pass (without --from-psi) "
              f"again so tab_bbl_cost_identified / tab_bbl_cbar / tab_bbl_cbar_design leave the "
              f"dead firm-quarters out as well !!!!")


def ridge_stats(eq: pd.DataFrame, dev: pd.DataFrame, is_B: bool | None = None,
                start: str | None = None, m: pd.DataFrame | None = None) -> dict:
    """Collinearity of the two columns that carry omega and zeta in the DIFFERENCED design.

    is_B selects a single firm type (True=B, False=D); None pools both, which is what the
    reported ridge table uses. The per-type call exists for check_psi_matches_solve(), because
    cost_params stores its ridge fields per type.

    start selects one launch quarter of a multi-start psi; None pools every quarter, which is
    what the pooled row of the by-start table and the whole of the per-routine ridge table use.

    m is the psi_dev x psi_eq merge when the caller already holds it (_merged), so a loop over
    quarters or types does not repeat the join.
    """
    if m is None:
        m = _merged(eq, dev)
    if is_B is not None:
        m = m[m["is_B"].to_numpy().astype(bool) == is_B]
    if start is not None and "start_q" in m.columns:
        m = m[m["start_q"].astype(str) == str(start)]
    d2 = (m["psi2_omega_e"] - m["psi2_omega_d"]).to_numpy(float)
    d4 = (m["psi4_zeta_e"] - m["psi4_zeta_d"]).to_numpy(float)
    # The same row filter as bbl_solve.rbar_of_block: the ratio is undefined where dpsi2 == 0,
    # i.e. where the deviation leaves the firm's discounted deposits unchanged. Those rows still
    # enter the eq:16 criterion (they are in the solve's n_rows), so the three counts are kept to
    # let the tables reconcile n here with the inequality count of tab_bbl_cost_identified. The
    # dead firm-quarters are not a filter of this function: the caller passes the rows the tables
    # describe (DEAD FIRM-QUARTERS in the module docstring).
    fin = np.isfinite(d2) & np.isfinite(d4)
    ok = fin & (d2 != 0)
    n_all, n_nonfinite, n_d2_zero = int(d2.size), int((~fin).sum()), int((fin & (d2 == 0)).sum())
    if ok.sum() < 2:
        raise ValueError(f"{int(ok.sum())} row(s) with dpsi2 != 0 -- no ratio statistics")
    d2, d4 = d2[ok], d4[ok]
    ratio = d4 / d2

    b = (d2 @ d4) / (d2 @ d2)                       # no-intercept fit: d4 ~ b*d2
    resid = d4 - b * d2
    r2 = 1.0 - (resid @ resid) / float(((d4 - d4.mean()) ** 2).sum())

    # Condition number, SCALE-FREE: each column is normalised to unit length before the SVD, so
    # the number reports collinearity alone (the Belsley-Kuh-Welsch condition index). On the raw
    # columns it would instead be ~1/(ratio*sqrt(1-R^2)), which folds in the ~38x units gap
    # between the columns (dpsi4 ~ rbar_f * dpsi2) — a property of the units, not of the
    # identification problem, and already reported in its own right as `ratio`. Not centered:
    # the design g = dpsi1 - omega*dpsi2 - gamma'dpsi3 - (1+zeta)*dpsi4 carries no intercept.
    X = np.column_stack([d2, d4])
    cond = float(np.linalg.cond(X / np.linalg.norm(X, axis=0)))

    # Across-Delta spread of the mean ratio: if a subsample were identified, this would be large.
    across = np.nan
    if "shock" in m.columns:
        g = pd.DataFrame({"s": m.loc[ok, "shock"].to_numpy(), "r": ratio}).groupby("s")["r"].mean()
        across = float(g.max() - g.min())

    # Robust and moment summaries of the ratio, side by side. The median is the statistic the
    # solve uses (per firm type it IS rbar_f); the mean and CV are kept because they are what a
    # reader computes first, with the share of the CV's sum of squares that its 10 most extreme
    # rows carry -- on the 2026-09-22 run 97.7% (E3) and 99.9% (E4), rows in which dpsi2 nearly
    # cancels, so the CV describes those rows and not the design.
    q25, med, q75 = (float(v) for v in np.quantile(ratio, [0.25, 0.5, 0.75]))
    dev_sq = (ratio - ratio.mean()) ** 2
    top = np.sort(dev_sq)[::-1][:10]
    return dict(
        n=int(len(d2)), n_all=n_all, n_nonfinite=n_nonfinite, n_d2_zero=n_d2_zero,
        n_firms=int(m["firm"].nunique()),
        n_shock=int(m.loc[ok, "shock"].nunique()) if "shock" in m.columns else -1,
        corr=float(np.corrcoef(d2, d4)[0, 1]),
        cond=cond,
        cond_raw=float(np.linalg.cond(X)),
        ratio_median=med, ratio_q25=q25, ratio_q75=q75,
        iqr_pct=float(100 * (q75 - q25) / abs(med)) if med else np.nan,
        ratio_mean=float(ratio.mean()),
        cv=float(100 * ratio.std() / abs(ratio.mean())),
        cv_top10_ss_pct=float(100 * top.sum() / dev_sq.sum()) if dev_sq.sum() > 0 else np.nan,
        one_minus_r2=float(1.0 - r2),
        across_delta=across,
        across_delta_pct=float(100 * across / ratio.mean()) if np.isfinite(across) else np.nan,
        n_starts=int(m["start_q"].nunique()) if "start_q" in m.columns else 0,
    )


def start_quarters(eq: pd.DataFrame) -> list:
    """The launch quarters a multi-start psi carries, in calendar order ([] when single-start).
    Sorted as STRINGS on purpose: the labels are 2016Q1..2024Q4, which sort calendar-correctly
    lexicographically, so no quarter parsing is introduced for a column that is already a label."""
    if "start_q" not in eq.columns:
        return []
    return sorted(str(q) for q in eq["start_q"].dropna().unique())


def ridge_by_start(eq: pd.DataFrame, dev: pd.DataFrame) -> dict:
    """-> {launch quarter: ridge_stats, 'pooled': ridge_stats}, or {} for a single-start psi.

    This is the table that decides whether the multi-start design bought anything: under one
    forward curve every quarter shares one median dpsi4/dpsi2 and the pooled ratio is a constant,
    so the corr/cond columns repeat down the page. Quarter-by-quarter movement in that median is
    the identifying variation, and it is visible here before it is visible in any estimate.
    """
    qs = start_quarters(eq)
    if not qs:
        return {}
    m = _merged(eq, dev)
    out = {}
    for q in qs:
        try:
            out[q] = ridge_stats(None, None, start=q, m=m)
        except (ValueError, ZeroDivisionError, FloatingPointError, IndexError) as exc:
            # One empty or degenerate quarter must not cost the other 35 their row.
            print(f"  [skip] start {q}: {type(exc).__name__}: {exc}")
    out["pooled"] = ridge_stats(None, None, m=m)
    return out


def load_cost_only() -> dict:
    """-> {routine: cost_params dict}, straight from cluster_processed, no psi involved.

    bbl_solve.py persists the design diagnostics it measured on ITS OWN psi
    (rbar_f, ridge_corr, ridge_cond, ridge_one_minus_R2, n_rows, n_firms). Everything both
    tables need is therefore already inside cost_params, and building from it alone makes a
    vintage mismatch impossible rather than merely detectable.
    """
    out = {}
    for E in ROUTINES:
        p = _cost_path(E, PSI_TAG)
        if p is not None:
            out[E] = json.loads(p.read_text(encoding="utf-8"))
            print(f"  read {p.name}")
        else:
            print(f"  [skip] E{E}: no cost_params_E{E}_spec_{SPEC}_{STAGE}"
                  f"{PSI_TAG if PSI_TAG is not None else '[_ms<S>]'}.json in {COST_DIR}")
    return out


def identified_split(cost: dict) -> bool:
    r"""Does the solve claim omega and zeta are SEPARATELY identified in this design?

    bbl_solve.py measures it -- the split is identified when rbar_f varies enough across the
    pooled launch quarters to break the Delta-psi_2 / Delta-psi_4 collinearity -- and records the
    verdict. This script only reads it, because the presentation and the table notes have to
    agree with whatever the solve concluded rather than assert a regime of their own.

    Three places are consulted, because the flag is a property of the SOLVE and could reasonably
    be written per routine or per firm-type block. The identified presentation fires when ANY
    block earned it; the blocks that did not are marked cell by cell (`_unidentified`), because
    one failing block must not erase the intervals the others carry, and neither branch's
    unqualified note is true of a mixed outcome. Measured 2026-09-22: 7 of 8 blocks pass the
    cond <= 30 gate and E4 x D sits at 33.4.
    """
    verdicts = []
    for E in sorted(cost):
        rec = cost[E] or {}
        for src in (rec.get("identified_split"),
                    (rec.get("_meta") or {}).get("identified_split")):
            if src is not None:
                verdicts.append(bool(src))
        for key, _lbl in BLOCKS:
            blk = rec.get(key)
            if not blk:
                continue
            # The block's own field, then the copy inside the ridge_by_start record it was
            # derived from. Both, because the flag is written in one place and copied to the
            # other, and a block carrying only one of them still states a verdict.
            for src in (blk.get("identified_split"),
                        (blk.get("by_start") or {}).get("identified_split")):
                if src is not None:
                    verdicts.append(bool(src))
                    break
    if not verdicts:
        return False
    if any(verdicts) and not all(verdicts):
        print("  [note] identified_split differs across blocks "
              f"({sum(verdicts)}/{len(verdicts)} true) — the identified presentation is used "
              "and each block that failed the gate is marked in the tables.")
    return any(verdicts)


def _cond_scale_free(corr: float) -> float:
    r"""Condition index of two unit-scaled columns from their correlation alone:
    kappa = sqrt((1+|r|)/(1-|r|)). Verified against the SVD on the psi parquets (271/295/654/278
    reproduced to the integer). Lets the scale-free index be reported from cost_params without
    the psi, since the solver persists ridge_corr but its ridge_cond is on the RAW columns and so
    also carries the ~1/rbar_f units gap."""
    r = min(abs(float(corr)), 1.0 - 1e-16)
    return float(np.sqrt((1.0 + r) / (1.0 - r)))


def collect_json(cost: dict) -> tuple[list, list]:
    """-> (ridge_blocks, rows) from cost_params only. `rows` matches collect()'s schema so
    build_identified() is reused unchanged; ridge_blocks is per (routine, firm type), which is
    the grain the solver actually measured on."""
    ridge_blocks, rows = [], []
    for E in sorted(cost):
        mode, dead_by = _null_fq_mode(cost[E]), _solve_dead_counts(cost[E])
        for key, lbl in BLOCKS:
            blk = cost[E].get(key)
            if not blk:
                continue
            w, z = float(blk["omega"]), float(blk["zeta"])
            rbar = float(blk.get("rbar_f", np.nan))
            ci = (blk.get("c_bar_ci_sqrtn") or {}).get("0.95") or {}
            corr = blk.get("ridge_corr")
            ridge_blocks.append(dict(
                E=E, block=key, label=lbl,
                n=blk.get("n_rows"), n_firms=blk.get("n_firms"),
                corr=corr,
                cond=_cond_scale_free(corr) if corr is not None else np.nan,
                cond_raw=blk.get("ridge_cond"),
                rbar=rbar, one_minus_r2=blk.get("ridge_one_minus_R2"),
                frac_bind=blk.get("frac_bind"),
                grad_inf=blk.get("opt_grad_inf"), ms_dF=blk.get("multistart_max_dF"),
            ))
            rows.append(dict(
                E=E, block=key, label=lbl, omega=w, zeta=z,
                omega_se=blk.get("omega_se"), zeta_se=blk.get("zeta_se"),
                gamma=blk.get("gamma") or {}, gamma_se=blk.get("gamma_se") or {},
                frac_bind=blk.get("frac_bind"), objective=blk.get("objective"),
                n=blk.get("n_rows"), n_firms=blk.get("n_firms"),
                # Of those inequalities, the rows with dpsi2 != 0 -- the only ones on which the
                # ratio, and so rbar_f, is defined, and the n the ridge tables count. Recorded by
                # the solve's pooled ridge_by_start; None on a cost_params without it.
                n_ratio=((blk.get("by_start") or {}).get("pooled") or {}).get("n_ratio"),
                rbar=rbar, cbar=float(blk.get("c_bar", w + rbar * z)),
                cbar_se=blk.get("c_bar_se_boot"),
                cbar_sd_rate=blk.get("c_bar_sd_rate_adj"),
                ci_lo=ci.get("lo"), ci_hi=ci.get("hi"),
                wz_corr=blk.get("omega_zeta_corr_boot"),
                ratio_wz=(-w / z) if z else np.nan,
                # The criterion inverted at the subsampled critical value (bbl_solve.py
                # --profile). These are the intervals the identified-split branch reports as
                # the estimand's uncertainty; they are absent from a solve run without
                # --profile, and the branch dashes them rather than falling back to the
                # bootstrap SD, which is not a valid reference for this criterion.
                ci_omega=blk.get("ci_omega"), ci_zeta=blk.get("ci_zeta"),
                by_start=blk.get("by_start") or {},
                # The solve's own verdict on this block's omega/zeta split and the pooled
                # condition index it was decided on. None on a cost_params written before the
                # gate existed -- which is "no verdict", not a failure.
                identified=blk.get("identified_split"),
                cond_pooled=(blk.get("by_start") or {}).get("pooled_bkw_cond"),
                corr_pooled=(blk.get("by_start") or {}).get("pooled_corr"),
                cond_max=(blk.get("by_start") or {}).get("ridge_cond_max"),
                # Dead firm-quarters (DEAD FIRM-QUARTERS in the module docstring): how this
                # routine's rows were filtered, the per-type counts, and -- when the default pass
                # restated a cost_params that predates the rule -- the rate the solve's c-bar
                # interval was evaluated at.
                null_mode=mode,
                n_null_fq=(dead_by.get(key) or {}).get("n_null_fq"),
                n_null_rows=(dead_by.get(key) or {}).get("n_null_rows"),
                ci_rbar=blk.get("_ci_rbar"),
            ))
    return ridge_blocks, rows


def check_psi_matches_solve(data: dict, rtol: float = 1e-6) -> list[str]:
    r"""Certify that the psi archive on disk is the SAME psi the solve consumed.

    Shape agreement is not enough. On 2026-08-17 an archive with byte-identical shapes to the
    logged run (307/306 firms, 15,100/15,050 rows) turned out to be a different vintage: the
    forward r^f curve had been rebuilt in between, so the psi VALUES moved ~1% while every
    dimension matched. Tables generated in that state pair a ridge diagnostic computed from one
    psi with cost parameters fitted on another.

    The test uses the fields bbl_solve.py persists from ITS OWN psi:

        rbar_f              = median(dpsi4/dpsi2)  -- contains no parameters at all
        ridge_corr          = corr(dpsi2, dpsi4)
        ridge_one_minus_R2  = 1 - R^2 of dpsi4 on dpsi2

    Being theta-free, they are pure functions of the design: if the psi matches, they reproduce
    exactly (measured: 0.0e+00), and any real difference is proof of a vintage mismatch rather
    than of a solver disagreement. Returns a list of human-readable mismatches (empty = clean).

    The comparison runs on the rows the SOLVE used (`psi_dev_solve`): without the dead
    firm-quarters when it dropped them, every row when it kept them or predates the rule. The
    record it is compared against is the solve's own (`cost_solve`), never a restated copy.
    The dead firm-quarter counts the solve recorded must equal the mask's, too (`null`).
    """
    problems, checked = [], 0
    for E, d in sorted(data.items()):
        nf = d.get("null") or {}
        for kappa, want in (nf.get("solve_counts") or {}).items():
            got = (nf.get("counts") or {}).get(kappa) or {}
            for f in ("n_null_fq", "n_null_rows"):
                checked += 1
                if want.get(f) is not None and want.get(f) != got.get(f):
                    problems.append(f"E{E}-{kappa} {f}: the psi has {got.get(f)} dead, the solve "
                                    f"recorded {want.get(f)} (rule tol {nf.get('tol'):g})")
        cost = d.get("cost_solve", d["cost"]) or {}
        dev = d.get("psi_dev_solve", d["psi_dev"])
        for key, _lbl in BLOCKS:
            blk = cost.get(key)
            if not blk:
                continue
            st = ridge_stats(d["psi_eq"], dev, is_B=(key == "B"))
            # Within one firm type the median ratio IS the solve's rbar_f.
            for field, mine in (("rbar_f", st["ratio_median"]),
                                ("ridge_corr", st["corr"]),
                                ("ridge_one_minus_R2", st["one_minus_r2"])):
                theirs = blk.get(field)
                if theirs is None:
                    continue                      # pre-dates the field; nothing to compare
                checked += 1
                if not np.isfinite(theirs) or not np.isfinite(mine):
                    continue
                denom = max(abs(float(theirs)), 1e-300)
                rel = abs(float(mine) - float(theirs)) / denom
                if rel > rtol:
                    problems.append(
                        f"E{E}-{key} {field}: psi gives {mine:.10g}, cost_params says "
                        f"{theirs:.10g} (rel {rel:.2e} > {rtol:g})")
    if not checked:
        problems.append("no theta-free ridge fields present in any cost_params — vintage "
                        "unverifiable; regenerate the solve so it persists rbar_f/ridge_corr")
    return problems


def collect(data: dict) -> tuple[dict, list]:
    ridge, rows = {}, []
    for E, d in sorted(data.items()):
        st = ridge_stats(d["psi_eq"], d["psi_dev"])
        st["n_shards"] = d["n_shards"]
        # The rows loaded and the dead firm-quarters taken out of them before `n_all`, so the
        # Notes can walk from the psi to n (_n_recon). Zero when nothing was dropped.
        nf = d.get("null") or {}
        st["null_mode"] = nf.get("mode", "none")
        st["n_loaded"] = int(nf.get("n_loaded", st["n_all"]))
        st["n_dead_rows"] = int(nf.get("dead_rows", 0)) if nf.get("dropped") else 0
        st["n_dead_fq"] = int(nf.get("dead_fq", 0)) if nf.get("dropped") else 0
        # The solve's inequality count for the routine (tab_bbl_cost_identified's n, B + D). The
        # psi merge must reproduce it row for row, or n here and n there are not the same design.
        n_ineq = [(d["cost"].get(k) or {}).get("n_rows") for k, _ in BLOCKS]
        st["n_ineq"] = (int(sum(n_ineq)) if all(v is not None for v in n_ineq) else None)
        if st["n_ineq"] is not None and st["n_ineq"] != st["n_all"]:
            print(f"  !!!! E{E}: the psi merge has {st['n_all']:,} rows but the solve counted "
                  f"{st['n_ineq']:,} inequalities -- the ridge n will not reconcile with "
                  f"tab_bbl_cost_identified !!!!")
        ridge[E] = st
        for key, lbl in BLOCKS:
            blk = d["cost"].get(key)
            if not blk:
                continue
            w, z = float(blk["omega"]), float(blk["zeta"])
            # Prefer the solve's own r_bar/c_bar (computed on the same design it fitted); fall
            # back to the psi-derived value for cost_params written before those fields existed.
            # The fallback is the median over THIS firm type's rows, which is how the solve
            # defines rbar_f -- never the routine's pooled median in `st`, which mixes B and D.
            rbar = blk.get("rbar_f")
            if rbar is None:
                rbar = ridge_stats(d["psi_eq"], d["psi_dev"], is_B=(key == "B"))["ratio_median"]
            rbar = float(rbar)
            ci = (blk.get("c_bar_ci_sqrtn") or {}).get("0.95") or {}
            rows.append(dict(
                E=E, block=key, label=lbl, omega=w, zeta=z,
                omega_se=blk.get("omega_se"), zeta_se=blk.get("zeta_se"),
                frac_bind=blk.get("frac_bind"), objective=blk.get("objective"),
                rbar=rbar, cbar=float(blk.get("c_bar", w + rbar * z)),
                cbar_se=blk.get("c_bar_se_boot"),
                cbar_sd_rate=blk.get("c_bar_sd_rate_adj"),
                ci_lo=ci.get("lo"), ci_hi=ci.get("hi"),
                wz_corr=blk.get("omega_zeta_corr_boot"),
                ratio_wz=(-w / z) if z else np.nan,
                identified=blk.get("identified_split"),
                cond_pooled=(blk.get("by_start") or {}).get("pooled_bkw_cond"),
                corr_pooled=(blk.get("by_start") or {}).get("pooled_corr"),
                cond_max=(blk.get("by_start") or {}).get("ridge_cond_max"),
            ))
    return ridge, rows


# ── number display (V_Main rule, 2026-09-28) ──────────────────────────────────
# Every displayed number, in the body and in the Notes, has exactly ND = 3 decimals; counts (n,
# firms, deviations, quarters, rows) are integers with thousands separators. A value that rounds to
# zero prints 0.000, never -0.000, and a NONZERO value that prints 0.000 is kept as it is (never
# rescaled) and recorded in ZERO_PRINTS, which each pass lists on stdout for the report.
ND = 3
ZERO_PRINTS: list = []


def _d3(x, where: str = "", nd: int = ND) -> str:
    """x at exactly `nd` decimals, thousands-separated (plain text), with the -0.000 guard."""
    v = float(x)
    s = f"{v:,.{nd}f}"
    if float(s.replace(",", "")) == 0.0:
        if v != 0.0 and where:
            ZERO_PRINTS.append((where, v))
        s = f"{0.0:.{nd}f}"
    return s


def _m3(x, where: str = "", nd: int = ND) -> str:
    """_d3 for math mode: a thousands separator in $...$ is punctuation and typesets as "1, 234",
    so it is braced ({,})."""
    return _d3(x, where, nd).replace(",", "{,}")


def _plain(label: str) -> str:
    r"""A row label without its LaTeX, for the ZERO_PRINTS list ('\hspace{1em}Basel Index ($t-1$)'
    -> 'Basel Index (t-1)')."""
    s = re.sub(r"\\hspace\{[^}]*\}|\\(?:hat|mathrm|boldsymbol|quad)\b|[${}\\]", "", label)
    return re.sub(r"\s+", " ", s).strip()


def _report_zero_prints(label: str) -> None:
    if ZERO_PRINTS:
        print(f"  [display] {label}: {len(ZERO_PRINTS)} nonzero value(s) print as 0.000 at "
              f"{ND} decimals (not rescaled):")
        for where, v in ZERO_PRINTS:
            print(f"      {where}: {v:.3e}")


# Rate LEVELS -- r-bar_f, c-bar, the forward-rate ratio Delta-psi_4/Delta-psi_2 -- are quarterly
# fractions in the solve and are shown as COMPOUNDED annual rates in percentage points,
# ((1 + x)^4 - 1) * 100, the convention of every other rate in the paper. Coefficients of the
# linear cost function (omega, zeta, gamma) cannot be compounded and stay per quarter. A standard
# deviation of a level is carried to the compounded scale by the delta method (_ann_sd); an
# interval by transforming its endpoints (the map is increasing for x > -1).
def _ann(x) -> float:
    return ((1.0 + float(x)) ** 4 - 1.0) * 100.0


def _ann_sd(x, sd) -> float:
    return 400.0 * (1.0 + float(x)) ** 3 * float(sd)


# ── latex helpers (V_Main house style) ──────────────────────────────────────
def _f(x, nd=ND, dash="---", where=""):
    return (dash if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"${_m3(x, where, nd)}$")


def _sci(x, nd=ND):
    r"""1.234e+15 -> $1.234\times10^{15}$, the mantissa at ND decimals. Python's own exponent form
    typesets as `1.23e - 05` inside math mode: the `e` italicises as a variable and the exponent's
    minus picks up binary-operator spacing."""
    m, e = f"{x:.{nd}e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def _num(x, nd=0):
    r"""Thousands separator that survives math mode: `$5,122$` typesets as `5, 122` because a
    comma is punctuation in math. Kept as text, so the digits stay upright like the other
    counts in these tables."""
    return f"{x:,.{nd}f}"


def _wrap(body, col_fmt, caption, label, header, footnote, ncols, nobreak=False):
    r"""Emit one \input-ready table in the V_Main house pattern (see polfunc_k4.tex, the sibling
    table in the same section): a \begin{spacing} group, xltabular sized to \textwidth, the
    caption and label INSIDE the table, repeat-header machinery, and the notes as the last
    footer row.

    xltabular rather than a `table` float + plain `tabular` for two reasons: X columns pin the
    table to the full text width (a plain tabular sizes to its content, and these nine numeric
    columns overrun the text block), and it breaks across pages instead of being pushed to the
    end of the section as an unsplittable float. The @{} pair on the notes row strips the edge
    padding so the note aligns with the rules above it.

    V_Main loads xltabular (L20), setspace (L28) and booktabs (L129) already.

    EVERY PAGE BREAK CARRIES THE CONTINUED HEAD AND FOOT. longtable reserves room for \endfoot
    only, and tests whether the last page can take \endlastfoot on the HEIGHT of that box. The
    notes row is a p-column, whose paragraph hangs below the row's first baseline, so the box is
    ~12pt high and ~120pt deep: the test passes, the notes are appended after the table has
    closed, and when they do not fit the ordinary output routine breaks the page between the last
    rows, with neither "Continued on next page" below nor "Table N (continued)" above. Measured in
    V_Main (tab_bbl_cbar_design, 2026-09-28) and in a V_Main-geometry harness at every filler
    offset from 216pt to 336pt. Two lines close it, both inside the table:
      * a zero kern after the notes row makes the last item of the \endlastfoot box a kern, so the
        box's depth moves into its height and longtable's own test sees the notes;
      * after the last body row, a kern of (lastfoot - foot) with a legal break right after it and
        an equal negative kern behind: the page builder must fit the last row TOGETHER with the
        room the notes will need, so when they do not fit it breaks before that row -- a normal
        longtable break, with the continued head and foot -- and the row moves over with the
        notes. The two kerns cancel, so a table that fits looks exactly as before. \nobreak ahead
        of them keeps the last row from being split from its own reservation.
    The notes stay in \endlastfoot: the look of the table does not change, only where it breaks.

    NON-BREAKING TABLES (`nobreak=True`, for a table shorter than a page: T8, T9). \\* on every row
    does not keep an xltabular together: the glue booktabs puts below each \midrule's rule is a
    legal page break whatever the row penalties are (measured 2026-09-29, 21 of 32 harness
    variants still split). The table is instead one unbreakable box -- a minipage holding the
    caption (\captionof, set as a top caption so its skip falls below it, as in the xltabular) and
    a tabularx with the same columns, rules, body and notes row -- between longtable's own \LTpre
    and \LTpost skips. It moves to the next page whole when it does not fit and looks the same.
    The line holding the box is centred on the table, so it leaves \prevdepth at half the table's
    height; the next paragraph's interline glue then collapses to \lineskip and setspace's closing
    -\baselineskip eats \LTpost (measured 2026-09-30: 1.8pt below the notes against 16.3pt after an
    xltabular). \prevdepth=0pt after the box reproduces the xltabular's ending, whose last foot has
    zero depth, so the following paragraph gets the same gap.
    """
    if nobreak:
        return "\n".join([
            r"\begin{spacing}{1.0}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}"
            r"\captionsetup{position=top}",
            rf"\captionof{{table}}{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{col_fmt}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            rf"\multicolumn{{{ncols}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
            rf"{{\footnotesize {footnote}}} \\",
            r"\end{tabularx}",
            r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
            r"\end{spacing}",
            "",
        ])
    return "\n".join([
        r"\begin{spacing}{1.0}",
        r"\footnotesize",
        # 8-9 numeric columns: at the 6pt default the inter-column padding alone eats ~2cm and
        # the widest cells ($0.99997278$, the 95% CI) overrun their X column. Measured: 3pt
        # clears every overfull box in both tables.
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{xltabular}}{{\textwidth}}{{{col_fmt}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}} \\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{ncols}}}{{c}}{{\bfseries Table \thetable\ (continued)}} \\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{ncols}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"\endfoot",
        r"\bottomrule",
        rf"\multicolumn{{{ncols}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {footnote}}} \\",
        LASTFOOT_KERN,
        r"\endlastfoot",
        *body,
        LASTROW_RESERVE,
        r"\end{xltabular}",
        r"\end{spacing}",
        "",
    ])


# The two lines _wrap adds so that every page break of an xltabular carries the continued head
# and foot (see its docstring). \csname...\endcsname reaches longtable's box registers without a
# \makeatletter, which a fragment \input mid-document cannot rely on. bbl_polfunc.py's tables use
# the same two lines.
LASTFOOT_KERN = r"\noalign{\kern0pt}"
LASTROW_RESERVE = (r"\noalign{\nobreak\dimen0=\dimexpr\ht\csname LT@lastfoot\endcsname"
                   r"+\dp\csname LT@lastfoot\endcsname-\ht\csname LT@foot\endcsname+2pt\relax"
                   r"\ifdim\dimen0<0pt \dimen0=0pt\fi\kern\dimen0\penalty9999\kern-\dimen0}")


def _small_cell(x, where=""):
    r"""A share such as 1-R^2 as a cell, at the three decimals of every number in these tables. A
    nonzero share below 5e-4 (the single-curve design sits at ~5e-5) prints 0.000 and is listed
    through ZERO_PRINTS rather than switched to another format."""
    if x is None or not np.isfinite(x):
        return "---"
    return _f(float(x), where=where)


def _n_recon(s, md: bool = False) -> str:
    r"""'223,200 $-$ 30,050 in 601 dead firm-quarters $=$ 193,150': the rows the psi holds, the
    rows of dead firm-quarters taken out, the rows the ratio drops (dpsi2 == 0 or non-finite) when
    there are any, and the n the ridge tables report. Built from the psi's own counts, so it holds
    by construction; collect() separately checks the rows left after the dead firm-quarters
    against the solve's n_rows."""
    minus, eq = (" - ", " = ") if md else (r" $-$ ", r" $=$ ")
    drop = s["n_d2_zero"] + s["n_nonfinite"]
    out = _num(s.get("n_loaded", s["n_all"]))
    if s.get("n_dead_rows"):
        out += f"{minus}{_num(s['n_dead_rows'])} in {_num(s['n_dead_fq'])} dead firm-quarters"
    if drop:
        out += (f"{minus}{_num(drop)}" + (" with dpsi_2 = 0" if md else r" with $\Delta\psi_2=0$")
                if s.get("n_dead_rows") else f"{minus}{_num(drop)}")
    return out + f"{eq}{_num(s['n'])}"


def _ridge_rows_note(ridge, md: bool = False) -> str:
    """The Notes sentence of tab_bbl_ridge_diagnostic that says what a row is and walks each
    routine's psi rows to its n (the dead firm-quarters, then the rows with dpsi2 == 0)."""
    Es = sorted(ridge)
    recon = "; ".join((f"E{E} " if md else rf"{rc.est_ref(E)} ") + _n_recon(ridge[E], md)
                      for E in Es)
    dead = any(ridge[E].get("n_dead_rows") for E in Es)
    same = all(ridge[E]["n"] == ridge[E].get("n_ineq") for E in Es)
    kept = [E for E in Es if ridge[E].get("null_mode") == "solve_keep"]
    if md:
        if dead:
            return ("A row is one firm × launch-quarter × deviation inequality outside the dead "
                    "firm-quarters, which are left out as in the parameter table (" + DEAD_FQ_DEF_MD
                    + "); n counts those with dpsi_2 != 0, on which dpsi_4/dpsi_2 is defined: "
                    + recon + (", the inequality count of the parameter table. " if same else
                               "; the rows with dpsi_2 = 0 still enter the inequality count of "
                               "the parameter table. "))
        return ("A row is one firm × launch-quarter × deviation inequality, firm types and "
                "quarters pooled. *n* counts only the rows with dpsi_2 != 0, on which "
                "dpsi_4/dpsi_2 is defined, so it is below the inequality count of the parameter "
                "table by the rows whose deviation leaves discounted deposits unchanged "
                f"(dpsi_2 = 0): {recon}. "
                + ("Dead firm-quarters are kept, as in the solve. " if kept else ""))
    # The table note states the rule and the n; the walk from psi rows to n (`recon`) stays in the
    # .md twin and on stdout (V_Main style guide, 2026-09-28).
    return (r"Rows: inequalities of \eqref{eq:16} with $\Delta\psi_2\neq0$, firm types and launch "
            r"quarters pooled"
            + (r", dead firm-quarters excluded" if dead
               else r", dead firm-quarters kept as in the solve" if kept else "")
            + (r"; $n$ is Table~\ref{tab:bbl_cost_identified}'s B $+$ D count. "
               if same else
               r"; $n$ falls short of the B $+$ D inequality count of "
               r"Table~\ref{tab:bbl_cost_identified} by the rows with $\Delta\psi_2=0$. "))


def _within_cond_min(blocks: dict | None):
    """(lowest per-launch-quarter condition index, its quarter) of one routine's ridge_by_start
    record, or (None, None) without one. The pooled record is not a quarter."""
    cs = [(s["cond"], q) for q, s in (blocks or {}).items()
          if q != "pooled" and s.get("cond") is not None and np.isfinite(s["cond"])]
    return min(cs) if cs else (None, None)


def build_ridge(ridge, disc="", by_start=None):
    # The trimmed ridge table (user, 2026-09-29): per routine, the pooled correlation and
    # condition index of [dpsi2 dpsi4], the lowest condition index within one launch quarter
    # when the psi is multi-start (`by_start`), and the median and IQR of dpsi4/dpsi2. The firm
    # and deviation counts are Table 7's; the mean, CV and 1-R^2 were dropped (the mean and CV
    # are carried by the ten most extreme rows, and 1-R^2 restates the correlation). The
    # sidecar still records every statistic of ridge_stats.
    # Headers are one short word each and every symbol is defined in the note. The condition
    # number is "Cond." and NOT $\kappa$ -- the paper already spends $\kappa$ on the firm type.
    within = {E: _within_cond_min((by_start or {}).get(E)) for E in ridge}
    has_within = any(v[0] is not None for v in within.values())
    X = r">{\centering\arraybackslash}X"
    ncol = 7 if has_within else 6
    col_fmt = r">{\raggedright\arraybackslash}p{1.25cm} " + " ".join([X] * (ncol - 1))
    if has_within:
        header = (r" & & & \multicolumn{2}{c}{Cond.} & \multicolumn{2}{c}{$\Delta\psi_4/\Delta\psi_2$}"
                  r" \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n"
                  r"Routine & $n$ & Corr & Pooled & Min.\ quarter & Median & IQR\%")
    else:
        header = (r" & & & & \multicolumn{2}{c}{$\Delta\psi_4/\Delta\psi_2$} \\ "
                  r"\cmidrule(lr){5-6}" + "\n"
                  r"Routine & $n$ & Corr & Cond. & Median & IQR\%")
    body = []
    for E in sorted(ridge):
        s = ridge[E]
        w = f"T8 {rc.est_ref(E)}"
        cells = [rc.est_ref(E), _num(s["n"]), _f(s["corr"], where=f"{w} Corr"),
                 _f(s["cond"], where=f"{w} Cond.")]
        if has_within:
            cells.append(_cond_cell(within[E][0], where=f"{w} Cond. min. quarter"))
        cells += [_f(_ann(s["ratio_median"]), where=f"{w} Median"),
                  _f(_iqr_ann(s), where=f"{w} IQR%")]
        body.append(" & ".join(cells) + r" \\")
    # Notes held to a few lines (V_Main style guide): the rows and n, then the column symbols. No
    # resampling is behind any cell here, so there are no draws or seed to state.
    foot = (
        r"\textit{Notes:} " + _ridge_rows_note(ridge)
        + r"Corr, Cond.: correlation and Belsley--Kuh--Welsch condition index of "
        r"$[\Delta\psi_2\;\Delta\psi_4]$ over the routine's rows"
        + (r"; Min.\ quarter: the lowest condition index within a single launch quarter"
           if has_within else "")
        + r". Median of the ratio in compounded annual percentage points (within a firm type, "
        r"the $\bar r^{f,\kappa}$ of Table~\ref{tab:bbl_cbar}); IQR\% relative to the median."
    )
    # Non-breaking (_wrap): the trimmed table is a few lines tall.
    return _wrap(body, col_fmt, r"$\psi_2/\psi_4$ Ridge",
                 "tab:bbl_ridge_diagnostic", header, _with(foot, disc), ncol, nobreak=True)


def _iqr_ann(s) -> float:
    """The interquartile range of the ratio as a percent of its median, both on the compounded
    annual scale the median is shown in (quartiles map through the increasing _ann)."""
    q25, med, q75 = (s.get(k) for k in ("ratio_q25", "ratio_median", "ratio_q75"))
    if None in (q25, med, q75):
        return s.get("iqr_pct", np.nan)
    m = _ann(med)
    return 100.0 * (_ann(q75) - _ann(q25)) / abs(m) if m else np.nan


def _cond_cell(x, sci_above=1e6, where=""):
    r"""A condition index as a table cell, at three decimals. Above ~1e6 it switches to
    $a\times10^{b}$ with a three-decimal mantissa: within a single launch quarter the two psi
    columns are collinear to machine precision (psi4 is rbar_f*psi2 there by construction), so the
    index runs to 1e15 and a thousands-separated number is 20 characters of noise in an X column
    sized for four."""
    if x is None or not np.isfinite(x):
        return "---"
    return _sci(float(x)) if abs(float(x)) >= sci_above else _f(float(x), where=where)


def _md_sci(x, nd=ND):
    """The markdown twin of _sci: 1.234e+15 -> 1.234×10^15."""
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}×10^{int(e)}"


def _md_cond(x, sci_above=1e6):
    """The markdown twin of _cond_cell: three decimals, a×10^b above 1e6."""
    if x is None or not np.isfinite(x):
        return "--"
    return _md_sci(float(x)) if abs(float(x)) >= sci_above else _d3(x)


def _md_small(x):
    """The markdown twin of _small_cell: three decimals."""
    if x is None or not np.isfinite(x):
        return "--"
    return _d3(x)


def _per_start(blk: dict) -> dict:
    """The launch-quarter map inside a cost_params block, whatever shape it was written in.

    bbl_solve.py stores ridge_by_start()'s whole return under blk['by_start'], so the quarters
    sit one level down under 'per_start' alongside the pooled record and the gate fields. A flat
    {quarter: record} map is accepted too, so a block written either way resolves here instead of
    at each of the two call sites.
    """
    bs = blk.get("by_start") or {}
    inner = bs.get("per_start")
    return inner if isinstance(inner, dict) else bs


def _solve_at(rec: dict, q: str | None, dead=()) -> tuple:
    r"""-> (frac_bind, c_bar) for one routine at one launch quarter, pooled across firm types.

    q=None takes the block's own pooled figures. A per-quarter figure exists only when the solve
    refitted inside the quarter (ridge_by_start's within-start refit); without that record the
    two columns are dashed rather than repeating the pooled number down 36 rows, which would read
    as 36 measurements of something that was measured once.

    Pooled across B and D by the inequality count, the same weight the criterion itself gives
    each block, so the pooled row equals what a single-block solve of the union would report if
    the two types shared a theta.

    `dead` names the firm types every firm-quarter of which is dead in quarter q (measured on the
    psi). Such a quarter has no fit for that type -- the solve that drops dead firm-quarters has no
    rows to refit, and one that kept them refitted rounding noise -- so both cells are dashed
    rather than reported for the other type alone under a "pooled" heading. A per-quarter record
    the default pass marked all_dead (_overlay_null_fq) counts the same way.
    """
    fb_num = cb_num = w_tot = 0.0
    seen = False
    for key, _lbl in BLOCKS:
        blk = (rec or {}).get(key)
        if q is not None and key in (dead or ()):
            return None, None
        if not blk:
            continue
        src = blk if q is None else _per_start(blk).get(q)
        if q is not None and (src or {}).get("all_dead"):
            return None, None
        if not src:
            continue
        fb, cb = src.get("frac_bind"), src.get("c_bar")
        if fb is None and cb is None:
            continue
        w = float(src.get("n_rows") or src.get("n") or blk.get("n_rows") or 1.0)
        seen = True
        w_tot += w
        fb_num += w * float(fb if fb is not None else np.nan)
        cb_num += w * float(cb if cb is not None else np.nan)
    if not seen or w_tot <= 0:
        return None, None
    return fb_num / w_tot, cb_num / w_tot


def _dead_cells(dead_q: dict, Es: list, md: bool = False) -> str:
    """'2016Q1 (D), in both panels': the launch quarters in which every firm-quarter of one type is
    dead, i.e. the rows of tab_bbl_ridge_by_start whose solve columns are dashed for that reason."""
    cells = {}
    for E in Es:
        for q, types in sorted(((dead_q or {}).get(E) or {}).items()):
            for k in types:
                cells.setdefault((q, k), []).append(E)
    parts = []
    for (q, k), where in sorted(cells.items()):
        if len(Es) > 1 and len(where) == len(Es):
            loc = ", in both panels" if len(Es) == 2 else ", in every panel"
        elif len(Es) > 1:
            loc = ", " + _join([(f"E{E}" if md else rf"Panel~{chr(65 + Es.index(E))}")
                                for E in where])
        else:
            loc = ""
        parts.append(f"{q} ({k}){loc}")
    return "; ".join(parts)


def _within_cond_range(by_start: dict, Es: list):
    """(min, max) of the per-quarter condition index over every panel, finite values only."""
    cs = [s["cond"] for E in Es for q, s in by_start[E].items()
          if q != "pooled" and s.get("cond") is not None and np.isfinite(s["cond"])]
    return (min(cs), max(cs)) if cs else None


def build_ridge_by_start(by_start: dict, cost: dict, disc="", dead_q=None, cbar=False):
    r"""The ridge measured one LAUNCH QUARTER at a time, one panel per routine.

    Separate from tab:bbl_ridge_diagnostic rather than added to it as extra rows: that table is
    the pooled statement (one row per routine) the text cites, and a reader checking whether the
    multi-start design broke the ridge is asking a different question -- does the median ratio
    MOVE across quarters -- which needs the quarters down the rows and the routines split into
    panels. The column is headed by the statistic, not by \bar r^f: that symbol is the per-type
    rate c-bar is evaluated at, and a quarter's median over both types is a different number.

    `dead_q` is {routine: {quarter: [firm types all of whose firm-quarters are dead there]}},
    measured on the psi; those rows have their solve columns dashed (_solve_at).

    `cbar` adds the per-quarter c-bar column (--per-quarter-cbar keep). Off by default (user
    decision 2026-09-28): within a quarter one forward curve prices every row, so omega and zeta
    are not separable there and the refit's c-bar says nothing the pooled fit does not.
    """
    col_fmt = (r">{\raggedright\arraybackslash}p{1.7cm} "
               rf"*{{{6 if cbar else 5}}}{{>{{\centering\arraybackslash}}X}}")
    header = (r"Start & $n$ & Median $\Delta\psi_4/\Delta\psi_2$ & Corr & Cond. & "
              r"Frac.\ viol." + (r" & $\bar c$" if cbar else ""))
    ncol = 7 if cbar else 6
    body = []
    Es = [E for E in ROUTINE_ORDER if E in by_start and by_start[E]]
    for pi, E in enumerate(Es):
        if pi:
            body.append(r"\midrule")
        if len(Es) > 1:
            body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Panel {chr(65 + pi)}: "
                        rf"{rc.est_ref(E)}}}}} \\")
            body.append(r"\addlinespace[0.3ex]")
        blocks = by_start[E]
        for q in [k for k in blocks if k != "pooled"] + ["pooled"]:
            s = blocks[q]
            fb, cb = _solve_at(cost.get(E) or {}, None if q == "pooled" else q,
                               ((dead_q or {}).get(E) or {}).get(q, ()))
            lbl = r"\textit{Pooled}" if q == "pooled" else q
            if q == "pooled":
                body.append(r"\addlinespace[0.3ex]")
            w = f"T10 {rc.est_ref(E)} {q}"
            body.append(" & ".join([
                lbl, _num(s["n"]), _f(_ann(s["ratio_median"]), where=f"{w} Median"),
                _f(s["corr"], where=f"{w} Corr"), _cond_cell(s["cond"], where=f"{w} Cond."),
                _f(fb, where=f"{w} Frac. viol."),
            ] + ([_f(_ann(cb) if cb is not None else None, where=f"{w} c-bar")]
                 if cbar else [])) + r" \\")
    modes = {_null_fq_mode(cost.get(E) or {}) for E in Es}
    dropped = bool(modes & {"solve_drop", "table_mask"})
    cells = _dead_cells(dead_q, Es)
    rng = _within_cond_range(by_start, Es) if cbar else None
    n_q = len([q for q in by_start[Es[0]] if q != "pooled"]) if Es else 0
    if rng and rng[1] < 1e6:
        rng_txt = f"{_d3(rng[0])} to {_d3(rng[1])}"
    elif rng:
        rng_txt = f"{_d3(rng[0])} and above"
    else:
        rng_txt = "in the hundreds and above"
    # Which cells are dashed, by provenance: a run whose solve predates the dead firm-quarter rule
    # refitted each quarter WITH them and stored no gamma, so none of its per-quarter shares can
    # be restated without them; a run whose solve dropped them dashes only the quarters with no
    # refit or with every firm-quarter of one type dead.
    if "table_mask" in modes:
        dash = (r"Per-quarter shares are dashed: this run's quarter refits kept the dead "
                r"firm-quarters and stored no $\boldsymbol{\gamma}$ to restate them without those "
                r"rows" + (r"; $\bar c$ is dashed where every firm-quarter of one type is dead ("
                           + cells + ")" if cbar and cells else "") + r". ")
    else:
        dash = (r"Dashes: no refit"
                + (r", or every firm-quarter of one type dead (" + cells + ")" if cells else "")
                + r". ")
    # Notes held to a few lines (V_Main style guide, 2026-09-28).
    foot = (
        r"\textit{Notes:} Table~\ref{tab:bbl_ridge_diagnostic}'s statistics by forward-curve "
        r"launch quarter, firm types pooled"
        + (r", dead firm-quarters excluded" if dropped else "")
        + r"; the quarters' $n$ sum to the \emph{Pooled} row. Movement in the median down a panel "
        r"is what separates $\omega$ from $\zeta$. \emph{Frac.\ viol.}: the last row of "
        r"Table~\ref{tab:bbl_cbar}, from a refit on the quarter alone"
        + (r" (with $\bar c$ at the quarter's median ratio)" if cbar else "")
        + r", pooled over firm types by inequality count (\emph{Pooled} row: the full-sample "
        r"fit). "
        + ((r"Per-quarter $\bar c$ is not separately informative: $\omega$ and $\zeta$ are not "
            r"separable within a quarter (condition index " + rng_txt + r"), and the pooled "
            r"estimate imposes one $(\omega,\zeta)$ across the " + str(n_q)
            + r" launch quarters. ") if cbar else "")
        + dash
    ).rstrip()
    return _wrap(body, col_fmt, r"$\psi_2/\psi_4$ Ridge by Forward-Curve Launch Quarter",
                 "tab:bbl_ridge_by_start", header, _with(foot, disc), ncol)


def md_ridge_by_start(by_start: dict, cost: dict, disc="", dead_q=None, cbar=False):
    L = []
    Es = [E for E in ROUTINE_ORDER if E in by_start and by_start[E]]
    for E in Es:
        blocks = by_start[E]
        L += [f"**E{E}**", "",
              "| Start | n | Median dpsi_4/dpsi_2 | Corr | Cond. | Frac. viol. |"
              + (" c-bar |" if cbar else ""),
              "|---|---:|---:|---:|---:|---:|" + ("---:|" if cbar else "")]
        for q in [k for k in blocks if k != "pooled"] + ["pooled"]:
            s = blocks[q]
            fb, cb = _solve_at(cost.get(E) or {}, None if q == "pooled" else q,
                               ((dead_q or {}).get(E) or {}).get(q, ()))
            fb_s = "--" if fb is None or not np.isfinite(fb) else _d3(fb)
            cb_s = "--" if cb is None or not np.isfinite(cb) else _d3(_ann(cb))
            L.append(f"| {'*pooled*' if q == 'pooled' else q} | {s['n']:,} | "
                     f"{_d3(_ann(s['ratio_median']))} | {_d3(s['corr'])} | "
                     f"{_md_cond(s['cond'])} | {fb_s} |" + (f" {cb_s} |" if cbar else ""))
        L.append("")
    modes = {_null_fq_mode(cost.get(E) or {}) for E in Es}
    dropped = bool(modes & {"solve_drop", "table_mask"})
    cells = _dead_cells(dead_q, Es, md=True)
    L += ["The differenced design measured per forward-curve launch quarter, firm types pooled. "
          "*n* counts the quarter's firm × launch-quarter × deviation rows with dpsi_2 != 0"
          + (" outside dead firm-quarters (defined with the parameter table)" if dropped else "")
          + ", as in the ridge "
          "table, so the quarters sum to the pooled row. *Median dpsi_4/dpsi_2* is the median "
          "row ratio, the forward rate a deviation's change in deposits is priced at, in "
          "compounded annual pp ((1+x)^4-1)*100 (as is c-bar); the pooled "
          "row takes it over all quarters and both firm types, while rbar_f^kappa (c-bar table) is "
          "the same median within one firm type. Movement in the median down a panel is the "
          "variation that separates omega from zeta. *Frac. viol.* (the share of deviation "
          "inequalities violated at the fit, as in the last row of the c-bar table)"
          + (" and *c-bar*" if cbar else "") + " come from the "
          "solve, pooled across firm types by the inequality count: per quarter from a refit on "
          "that quarter alone"
          + (" (c-bar at the quarter's own median ratio per type, which one forward curve pins "
             "although it does not split omega from zeta)" if cbar else "")
          + ", dashed where undefined"
          + (", including a firm type all of whose firm-quarters in the quarter are dead: "
             + cells if cells else "")
          + "; in the pooled row from the full-sample fit."
          + (" For this run the per-quarter refits kept the dead firm-quarters and did not store "
             "their gamma, so their violation share is dashed." if "table_mask" in modes else "")
          + (" " + disc if disc else "")]
    return "\n".join(L)


# ── the violated share by deviation direction (appendix table) ─────────────────
# The deviation grid applies every magnitude once raising and once lowering the firm's choice
# spread (bbl_fwd_sim.jl deviation_shifts: shifts = [+g1, -g1, +g2, -g2, ...] and
# sigma-tilde = sigma-hat + shifts[s], so under dev-scheme=grid an odd global index `shock` RAISES
# the spread). A psi_dev that carries `shift_pp` (Delta_s in compounded annual pp) states the
# direction itself and is preferred. The parity is checked on the data before it is used: raising
# the spread raises the spread revenue psi1, so dpsi1 = psi1_eq - psi1_dev is negative on most
# upward rows and positive on most downward ones (measured on _ms1: 0.900 of the rows under E3 and
# 0.958 under E4; the two members of a +/- pair have opposite dpsi1 signs in 0.968 and 0.990 of the
# pairs). dpsi2 is no guide to the direction: at large magnitudes it has the same sign both ways.
PARITY_MIN_AGREE = 0.75


def _deviation_up(m: pd.DataFrame):
    """-> (upward mask over the rows of a psi_dev x psi_eq merge, how it was decided), or
    (None, why not)."""
    if "shift_pp" in m.columns:
        sp = pd.to_numeric(m["shift_pp"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(sp)) or np.any(sp == 0.0):
            return None, "shift_pp has zero or non-finite values"
        return sp > 0.0, "the sign of shift_pp"
    if "shock" not in m.columns:
        return None, "psi_dev carries neither shift_pp nor shock"
    up = (m["shock"].to_numpy().astype(int) % 2) == 1
    d1 = (m["psi1_e"] - m["psi1_d"]).to_numpy(float)
    agree = float(np.mean(np.where(up, d1 < 0.0, d1 > 0.0)))
    if not agree >= PARITY_MIN_AGREE:
        return None, (f"the parity of shock is contradicted by the data (the sign of dpsi1 agrees "
                      f"on {agree:.3f} of the rows, below {PARITY_MIN_AGREE})")
    return up, (f"the parity of shock under dev-scheme=grid (odd raises the spread; the sign of "
                f"dpsi1 agrees on {agree:.3f} of the rows)")


def violated_by_sign(data: dict) -> dict:
    """{routine: {kappa: dict(up, down, all, n_up, n_down)}}: the share of inequalities with g < 0
    at the solve's theta-hat, over the rows the other tables describe (dead firm-quarters out
    whenever the solve or the tables drop them), split by the direction of the deviation. The
    pooled share must equal the violation share the cost tables print; a difference is reported."""
    out = {}
    for E, d in sorted(data.items()):
        m = _merged(d["psi_eq"], d["psi_dev"])
        up, how = _deviation_up(m)
        if up is None:
            print(f"  !!!! [up/down] E{E}: direction unknown -- {how}; the routine is left out of "
                  f"tab_bbl_violated_by_sign !!!!")
            continue
        print(f"  [up/down] E{E}: direction from {how}")
        isb = m["is_B"].to_numpy().astype(bool)
        for kappa, mk in (("B", isb), ("D", ~isb)):
            blk = (d["cost"] or {}).get(kappa)
            if not blk or not mk.any():
                continue
            g = _g_rows(m.loc[mk], blk)
            u = up[mk]
            rec = dict(up=float(np.mean(g[u] < 0.0)) if u.any() else np.nan,
                       down=float(np.mean(g[~u] < 0.0)) if (~u).any() else np.nan,
                       all=float(np.mean(g < 0.0)), n_up=int(u.sum()), n_down=int((~u).sum()))
            out.setdefault(E, {})[kappa] = rec
            fb = blk.get("frac_bind")
            if fb is not None and abs(rec["all"] - float(fb)) > 1e-9:
                print(f"  !!!! [up/down] E{E}-{kappa}: pooled share {rec['all']:.6f} differs from "
                      f"the violation share of the cost tables, {float(fb):.6f} !!!!")
            print(f"  [up/down] E{E}-{kappa}: up {rec['up']:.4f} (n {rec['n_up']:,}), down "
                  f"{rec['down']:.4f} (n {rec['n_down']:,}), all {rec['all']:.4f}")
    return out


def build_violated_by_sign(res: dict, disc="", dead=False):
    r"""APPENDIX table: the last row of tab_bbl_cbar split by the direction of the deviation, one
    column per routine x firm type (routines spanning their two types, as in tab_bbl_cbar_design).
    An upward deviation raises the firm's spread (pays depositors less); a violated upward
    inequality says that paying less would have raised the firm's value."""
    Es = [E for E in ROUTINE_ORDER if E in res]
    cols = [(E, k) for E in Es for k, _ in BLOCKS if k in res[E]]
    ncol = 1 + len(cols)
    col_fmt = (r">{\raggedright\arraybackslash}p{4.4cm} "
               rf"*{{{len(cols)}}}{{>{{\centering\arraybackslash}}X}}")
    spans, rules, c0 = [], [], 2
    for E in Es:
        nk = sum(1 for k, _ in BLOCKS if k in res[E])
        spans.append(rf"\multicolumn{{{nk}}}{{c}}{{{rc.est_ref(E)}}}")
        rules.append(rf"\cmidrule(lr){{{c0}-{c0 + nk - 1}}}")
        c0 += nk
    header = (" & " + " & ".join(spans) + r" \\ " + "".join(rules) + "\n"
              + " & " + " & ".join(k for _E, k in cols))

    def row(label, f):
        return label + " & " + " & ".join(f(res[E][k], E, k) for E, k in cols) + r" \\"

    def share(key):
        return lambda r, E, k: _f(r[key], where=f"T-updown {rc.est_ref(E)} {k} {key}")

    body = [row(r"Upward deviations ($\Delta>0$)", share("up")),
            row(r"Downward deviations ($\Delta<0$)", share("down")),
            r"\addlinespace[0.3ex]",
            row(r"All deviations", share("all")),
            r"\midrule"]
    if all(r["n_up"] == r["n_down"] for E in Es for r in res[E].values()):
        body.append(row(r"Inequalities $n$ per direction", lambda r, E, k: _num(r["n_up"])))
    else:
        body.append(row(r"Inequalities $n$, upward", lambda r, E, k: _num(r["n_up"])))
        body.append(row(r"Inequalities $n$, downward", lambda r, E, k: _num(r["n_down"])))
    foot = (
        r"\textit{Notes:} The share of the inequalities of \eqref{eq:16} failing at the solve's "
        r"$\hat\theta$ (the last row of Table~\ref{tab:bbl_cbar}), by the direction of the "
        r"deviation: an upward deviation raises the firm's spread by $\Delta>0$, paying depositors "
        r"less, and a downward one lowers it; every magnitude is applied in both directions"
        + (r", and dead firm-quarters are excluded" if dead else "") + r"."
    )
    return _wrap(body, col_fmt, r"BBL Violated Inequalities by Deviation Direction",
                 "tab:bbl_violated_by_sign", header, _with(foot, disc), ncol)


def md_violated_by_sign(res: dict, disc="", dead=False):
    Es = [E for E in ROUTINE_ORDER if E in res]
    cols = [(E, k) for E in Es for k, _ in BLOCKS if k in res[E]]
    L = ["| | " + " | ".join(f"E{E} {k}" for E, k in cols) + " |",
         "|---|" + "---:|" * len(cols)]
    for lab, key in (("upward (Delta > 0)", "up"), ("downward (Delta < 0)", "down"),
                     ("all deviations", "all")):
        L.append(f"| {lab} | " + " | ".join(_d3(res[E][k][key]) for E, k in cols) + " |")
    L.append("| n upward / downward | "
             + " | ".join(f"{res[E][k]['n_up']:,} / {res[E][k]['n_down']:,}" for E, k in cols)
             + " |")
    L += ["", "Share of the eq:16 inequalities that fail at the solve's theta-hat (the last row of "
              "the c-bar table), by the direction of the deviation: upward raises the firm's spread "
              "by Delta > 0 (pays depositors less), downward lowers it; every magnitude is applied "
              "in both directions" + (", and dead firm-quarters are excluded" if dead else "")
              + "." + (" " + disc if disc else "")]
    return "\n".join(L)


# Routine labels in paper order; \ref renders these as the Roman numerals of V_Main's
# enumerate(label=(\Roman*)), which is how every other estimates table heads its columns.
ROUTINE_ORDER = tuple(_routines.LINK_ESTS)
# Display names follow the policy-function tables (polfunc_k4/k5), the only house precedent for
# these four variables. gamma has never been reported before, so there is nothing else to match.
Z_LABELS = [
    ("psi3_gamma_personnel_cost_ratio_lag", r"Personnel Cost Ratio ($t-1$)"),
    ("psi3_gamma_admin_cost_ratio_lag",     r"Admin Cost Ratio ($t-1$)"),
    ("psi3_gamma_tax_cost_ratio_lag",       r"Tax Cost Ratio ($t-1$)"),
    ("psi3_gamma_indice_basileia_lag",      r"Basel Index ($t-1$), bp per pp"),
]
# The Basel loading and its SD are 1e-6 to 1e-3 of quarterly marginal cost per percentage point of
# the index, so at three decimals they print 0.000. Both are shown in BASIS POINTS of quarterly
# marginal cost per pp of the index (user decision 2026-09-29), x10^4 on the native per-quarter
# fraction, with the unit in the row label. Every other coefficient keeps its native units.
Z_SCALE = {"psi3_gamma_indice_basileia_lag": 1e4}


def _zscaled(v, key):
    """A gamma estimate or SD in its display unit (Z_SCALE); None passes through."""
    return None if v is None else float(v) * Z_SCALE.get(key, 1.0)


def _stars(est, se):
    r"""House significance markers from the normal approximation to the bootstrap SD:
    *** p<0.01, ** p<0.05, * p<0.1 (polfunc_k4.tex convention, which renders `^{}` when none).

    Reported because the estimates table conventionally carries them, but see the table note:
    the eq:16 criterion is kinked and potentially set-identified, so a normal approximation is
    not the right reference distribution and these should not be read as tests.
    """
    if est is None or se is None or not np.isfinite(est) or not np.isfinite(se) or se <= 0:
        return ""
    t = abs(est / se)
    return "***" if t >= 2.576 else "**" if t >= 1.96 else "*" if t >= 1.645 else ""


def _unidentified(rows):
    """Blocks whose solve said the omega/zeta split is NOT identified, as (E, kappa, cond).
    Explicit False only: a row with no verdict (a pre-gate cost_params) is not a failure."""
    return [(r["E"], r["block"], r.get("cond_pooled"))
            for r in rows if r.get("identified") is False]


def _fmt_cond(c):
    return _d3(c) if isinstance(c, (int, float)) and np.isfinite(c) else "n/a"


def _unidentified_note(rows):
    """LaTeX clause for the Notes: which blocks are marked and on what statistic, or the
    statement that every block cleared the gate. Read off the rows, never asserted."""
    thr = next((r.get("cond_max") for r in rows if r.get("cond_max")), 30.0)
    bad = _unidentified(rows)
    if not bad:
        worst = max((r.get("cond_pooled") or 0.0) for r in rows)
        return (rf"every block clears the threshold of {thr:.0f} on the pooled "
                rf"Belsley--Kuh--Welsch condition index of $[\Delta\psi_2\;\Delta\psi_4]$ "
                rf"(row \emph{{Condition index}}; largest {_fmt_cond(worst)})")
    lst = "; ".join(rf"{rc.est_ref(E)}, {k}: {_fmt_cond(c)}" for E, k, c in bad)
    return (rf"identification is assessed per block by the pooled Belsley--Kuh--Welsch "
            rf"condition index of $[\Delta\psi_2\;\Delta\psi_4]$ (row \emph{{Condition index}}, "
            rf"the columns scaled to unit length) against a threshold of "
            rf"{thr:.0f}; the block(s) marked $\ddagger$ exceed it ({lst}), so there "
            rf"$\omega^\kappa$ and $\zeta^\kappa$ are not separately identified and only "
            rf"$\bar c^\kappa$ is interpretable")


def _md_unident(rows):
    bad = _unidentified(rows)
    if not bad:
        return ""
    lst = "; ".join(f"E{E} {k}: {_fmt_cond(c)}" for E, k, c in bad)
    return (" ‡ marks a block whose pooled condition index exceeds the threshold "
            f"({lst}); there omega and zeta are not separately identified and only c-bar "
            "is interpretable.")


def _md_unident_ref(rows):
    """'the ‡ block of the parameter table, E4 D: 33.4' -- for a table that carries no ‡ marks of
    its own and so has to point to where they are."""
    bad = _unidentified(rows)
    lst = "; ".join(f"E{E} {k}: {_fmt_cond(c)}" for E, k, c in bad)
    return f"the ‡ block{'s' if len(bad) > 1 else ''} of the parameter table, {lst}"


def _corr_clause(rows):
    """The measured pooled correlation for the ridge-degenerate note; the historical
    figure only when the solve recorded none."""
    cs = [r.get("corr_pooled") for r in rows if r.get("corr_pooled") is not None]
    return (rf"$\mathrm{{corr}}\ge{_m3(min(cs))}$" if cs else r"$\mathrm{corr}>0.999$")


def _n_pairs(rows):
    """-> [(E, n_B, n_D, n_ratio_B, n_ratio_D)] for every routine whose two blocks both carry
    the inequality count and the dpsi2 != 0 count; [] when any is missing (a cost_params written
    before the solve recorded n_ratio), in which case the notes give the definition alone."""
    by = {(r["E"], r["block"]): r for r in rows}
    out = []
    for E in [E for E in ROUTINE_ORDER if any((E, k) in by for k, _ in BLOCKS)]:
        b, d = by.get((E, "B")), by.get((E, "D"))
        if not (b and d):
            return []
        vals = (b.get("n"), d.get("n"), b.get("n_ratio"), d.get("n_ratio"))
        if any(v is None for v in vals):
            return []
        out.append((E, *(int(v) for v in vals)))
    return out


def _dead_by_E(rows) -> dict:
    """{routine: {kappa: (n_null_fq, n_null_rows)}} for the routines whose dead firm-quarters are
    out of every figure (the solve dropped them, or the default pass restated it without them)."""
    out = {}
    for r in rows:
        if r.get("null_mode") in ("solve_drop", "table_mask") and r.get("n_null_fq") is not None:
            out.setdefault(r["E"], {})[r["block"]] = (int(r["n_null_fq"]),
                                                      int(r.get("n_null_rows") or 0))
    return out


def _dead_list(rows, md: bool = False, with_rows: bool = True, short: bool = False) -> str:
    """'(III): 412 B and 189 D firm-quarters, 20,600 and 9,450 rows; (IV): ...'; `short` gives
    '(III): 412 B, 189 D; (IV): 410 B, 185 D'."""
    by = _dead_by_E(rows)
    parts = []
    for E in [E for E in ROUTINE_ORDER if E in by]:
        d = by[E]
        ks = [k for k, _ in BLOCKS if k in d]
        if short:
            parts.append(_join([f"{_num(d[k][0])} {k}" for k in ks]) + " under "
                         + (f"E{E}" if md else rc.est_ref(E)))
            continue
        s = ((f"E{E}" if md else rc.est_ref(E)) + ": "
             + _join([f"{_num(d[k][0])} {k}" for k in ks]) + " firm-quarters")
        if with_rows:
            s += ", " + _join([_num(d[k][1]) for k in ks]) + " rows"
        parts.append(s)
    return "; ".join(parts)


def _dead_note(rows, md: bool = False) -> str:
    """The Notes clause of the parameter table that defines a dead firm-quarter and counts them;
    the other tables point to it (_dead_ref). Empty when no routine's rows were filtered."""
    by = _dead_by_E(rows)
    if not by:
        kept = any(r.get("null_mode") == "solve_keep" for r in rows)
        return ("Dead firm-quarters are kept, as in the solve. " if kept else "")
    mask = any(r.get("null_mode") == "table_mask" for r in rows if r["E"] in by)
    if md:
        return ("Dead firm-quarters are left out of every count, share and rate ("
                + _dead_list(rows, md=True) + "): " + DEAD_FQ_DEF_MD + ". "
                + ("This run's solve predates that rule and kept them; they are found here in its "
                   "psi, and the estimates are the solve's, on which those rows carry no weight. "
                   if mask else ""))
    # Conditional on provenance: a solve that applied the rule dropped them itself; a cost_params
    # written before the rule was solved with them, and only the tables leave them out.
    # The user's wording (2026-09-28). The counts per type and routine are in the .md twin and on
    # stdout; the note stays at its few lines.
    if mask:
        return (r"Dead firm-quarters (no deposits of the deviated type) are excluded from every "
                r"count and share here; this run's solve retained them. ")
    return (r"Dead firm-quarters, " + DEAD_FQ_DEF + r", are excluded from the solve and from "
            r"every count. ")


def _dead_ref(rows, md: bool = False) -> str:
    """Short form for the tables after the parameter table: where the definition and the count are."""
    by = _dead_by_E(rows)
    if not by:
        return ""
    return (("Dead firm-quarters (defined with the parameter table) are left out: "
             + _dead_list(rows, md=True, with_rows=False) + ". ") if md else
            r"dead firm-quarters (Table~\ref{tab:bbl_cost_identified}) excluded")


def _md_n_ineq(rows):
    pairs = _n_pairs(rows)
    dead = bool(_dead_by_E(rows))
    s = ("Inequalities n counts every firm × launch-quarter × deviation row of the block"
         + (" outside dead firm-quarters" if dead else "") + ", all of which enter eq:16")
    if pairs and all(rb == nb and rd == nd for _E, nb, nd, rb, rd in pairs):
        return s + "; every one has dpsi_2 != 0, so the ridge tables count the same rows."
    s += ("; the ridge tables count only the rows with dpsi_2 != 0, on which dpsi_4/dpsi_2 is "
          "defined")
    if not pairs:
        return s + "."
    return (s + " (B + D: " + "; ".join(f"E{E} {rb:,} + {rd:,} = {rb + rd:,} of {nb + nd:,}"
                                        for E, nb, nd, rb, rd in pairs)
            + "). The rows left out are deviations that leave the firm's discounted deposits "
              "unchanged (dpsi_2 = 0).")


# The violation share reads as "no information about fit" only when it sits at its mechanical
# value: a symmetric +/- grid around a policy with no interior turning point fails exactly one of
# each pair for ANY theta. The band is bbl_solve.py's own warning band (0.45 <= frac_bind <= 0.55).
# Measured on the 2026-09-22 run: with the dead firm-quarters counted the B shares were 0.401 and
# 0.410, outside it, because their rows are g = 0 up to rounding and so never "fail"; without
# them all four shares are 0.471-0.484.
MECH_BAND = 0.05
_MECH_TEX = (r"all within $0.050$ of the mechanical $1/2$, so the row carries no information "
             r"about fit")


# The c-bar panel rows shared by tab_bbl_cbar and tab_bbl_cbar_design: the two rates compounded to
# annual percentage points, the bootstrap SD carried over by the delta method, the interval by its
# endpoints, the violation share as it is -- every number at three decimals.
def _w(r, what: str, table: str) -> str:
    return f"{table} {rc.est_ref(r['E'])} {r['block']} {what}"


def _cell_rbar(r, table):
    return f"${_m3(_ann(r['rbar']), _w(r, 'rbar', table))}$"


def _cell_cbar(r, table):
    return f"${_m3(_ann(r['cbar']), _w(r, 'c-bar', table))}$"


def _cell_cbar_sd(r, table):
    if r.get("cbar_se") is None:
        return ""
    return f"$({_m3(_ann_sd(r['cbar'], r['cbar_se']), _w(r, 'c-bar SD', table))})$"


def _cell_cbar_ci(r, table):
    if r.get("ci_lo") is None:
        return "---"
    return (rf"{{\scriptsize $[{_m3(_ann(r['ci_lo']), _w(r, 'CI lo', table))},"
            rf"{_m3(_ann(r['ci_hi']), _w(r, 'CI hi', table))}]$}}")


def _cell_share(r, table):
    if r.get("frac_bind") is None:
        return "---"
    return f"${_m3(r['frac_bind'], _w(r, 'violated share', table))}$"


_ANN_TEX = r"Rates in compounded annual percentage points, $((1+x)^4-1)\times100$. "


def _mechanical(rows) -> bool:
    fbs = [float(r["frac_bind"]) for r in rows if r.get("frac_bind") is not None]
    return bool(fbs) and all(abs(f - 0.5) <= MECH_BAND for f in fbs)


def _ci_rbar_note(rows, md: bool = False) -> str:
    """For a cost_params the default pass restated without the dead firm-quarters: the interval is
    the solve's, evaluated at the solve's rbar_f, and says so, with that rate and how far the
    point estimate moved. Empty for every other run."""
    rs = [r for r in rows if r.get("ci_rbar") is not None]
    if not rs:
        return ""
    # How far c-bar moves between the two rates, on the compounded annual scale it is shown on.
    shift = max(abs(_ann(r["cbar"]) - _ann(r["omega"] + float(r["ci_rbar"]) * r["zeta"]))
                for r in rs)
    Es = [E for E in ROUTINE_ORDER if any(r["E"] == E for r in rs)]
    lst = "; ".join(
        (f"E{E} " if md else rf"{rc.est_ref(E)} ")
        + ", ".join((f"{r['block']} {_d3(_ann(r['ci_rbar']))} pp" if md else
                     rf"{r['block']} ${_m3(_ann(r['ci_rbar']))}$")
                    for r in rs if r["E"] == E) for E in Es)
    if md:
        return ("For this run the interval is the solve's, evaluated at rbar_f over every row, "
                f"dead firm-quarters included ({lst}); its subsampling draws are not stored, so it "
                "is not re-evaluated at the rate shown, which moves c-bar by at most "
                f"{_d3(shift)} pp. ")
    # One clause in the table (style guide): the per-block rates stay in the .md twin.
    return (r"(here at the $\bar r^{f,\kappa}$ of all rows; $\hat{\bar c}^\kappa$ moves by at "
            rf"most ${_m3(shift)}$)")


def build_cbar_panels(rows, identified=False, disc=""):
    r"""\bar c^\kappa and the health metric that governs whether it may be read.

    Split out of the parameter table deliberately: frac_bind is the gate on quoting any cost
    figure at all, so it belongs on one page with the cost figure rather than buried under eight
    rows of coefficients.

    `identified` says whether the solve separated \omega from \zeta. It changes what \bar c IS,
    not how it is computed: with a single forward curve it is the one direction the design pins
    down and therefore the estimand; with launch-quarter variation \omega and \zeta are each
    identified and \bar c restates them at \bar r^{f,\kappa}, except in a block marked \ddagger,
    where the split failed the gate and \bar c is again the interpretable object. The notes say
    which, because the unqualified claim is false in the other regime.
    """
    by = {(r["E"], r["block"]): r for r in rows}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in by for k, _ in BLOCKS)]
    ncol = 1 + len(Es)
    col_fmt = (r">{\raggedright\arraybackslash}p{5.0cm} "
               rf"*{{{len(Es)}}}{{>{{\centering\arraybackslash}}X}}")
    header = " & " + " & ".join(rc.est_ref(E) for E in Es)
    body = []
    for pi, (kappa, _lbl) in enumerate(BLOCKS):
        if not any((E, kappa) in by for E in Es):
            continue
        title = ("Brick-and-Mortar (B) Firms" if kappa == "B" else "Digital (D) Firms")
        if pi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Panel {'AB'[pi]}: {title}}}}} \\")
        body.append(r"\addlinespace[0.3ex]")

        def cells(f, dash="---"):
            out = []
            for E in Es:
                r = by.get((E, kappa))
                out.append(f(r) if r else dash)
            return " & ".join(out)

        body.append(_rbar_label(kappa) + " & "
                    + cells(lambda r: _cell_rbar(r, "T9")) + r" \\")
        body.append(r"\addlinespace[0.3ex]")
        body.append(rf"$\hat{{\bar c}}^{{\mathrm{{{kappa}}}}}$ & "
                    + cells(lambda r: _cell_cbar(r, "T9")) + r" \\")
        body.append(" & " + cells(lambda r: _cell_cbar_sd(r, "T9")) + r" \\")
        body.append(r"\quad 95\% CI & " + cells(lambda r: _cell_cbar_ci(r, "T9")) + r" \\")
        body.append(r"\addlinespace[0.4ex]")
        body.append(FRAC_BIND_LABEL + " & " + cells(lambda r: _cell_share(r, "T9")) + r" \\")
    # Notes held to a few lines (V_Main style guide, 2026-09-28): what c-bar and rbar_f are (the
    # one place rbar_f is defined), their compounded annual units, the regime in one clause, what
    # the parentheses and brackets hold, the last row, and the discount sentence.
    ci_note = _ci_rbar_note(rows)
    if identified and _unidentified(rows):
        regime = (r"; in the $\ddagger$ block of Table~\ref{tab:bbl_cost_identified} it is the "
                  r"only interpretable cost object. ")
    elif identified:
        regime = r". "
    else:
        regime = (r"; it is the only cost object this design identifies ("
                  + _corr_clause(rows) + r", Table~\ref{tab:bbl_ridge_diagnostic}). ")
    foot = (
        r"\textit{Notes:} $\bar c^\kappa=\omega^\kappa+\bar r^{f,\kappa}\zeta^\kappa$, excluding "
        r"$\boldsymbol{\gamma}'\boldsymbol{Z}$, where "
        + _rbar_def(dead_ref="dead firm-quarters excluded" if _dead_by_E(rows) else "")
        + regime + _ANN_TEX
        + r"Parentheses: bootstrap standard deviation (delta method); brackets: subsampling 95\% "
        r"interval" + (" " + ci_note if ci_note else "") + r". "
        r"Last row: share of the inequalities of \eqref{eq:16} failing at $\hat\theta$"
        + (r", " + _MECH_TEX if (not identified or _mechanical(rows)) else "") + r"."
    )
    # One caption for both regimes. c-bar is identified whether or not the omega/zeta split is,
    # and the violation-share row is the criterion health; the Notes carry the regime.
    caption = r"BBL Identified Marginal Cost $\bar c^\kappa$ and Criterion Health"
    # Non-breaking (_wrap): the table is well under a page, and a break left its Panel B header
    # alone at the foot of a page.
    return _wrap(body, col_fmt, caption, "tab:bbl_cbar", header, _with(foot, disc), ncol,
                 nobreak=True)


def md_cbar_panels(rows, identified=False, disc=""):
    by = {(r["E"], r["block"]): r for r in rows}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in by for k, _ in BLOCKS)]
    L = ["| | " + " | ".join(f"E{E}" for E in Es) + " |", "|---|" + "---:|" * len(Es)]
    for kappa, _ in BLOCKS:
        if not any((E, kappa) in by for E in Es):
            continue
        L.append(f"| **Panel {'A' if kappa=='B' else 'B'}: "
                 f"{'Brick-and-Mortar (B)' if kappa=='B' else 'Digital (D)'} Firms** | "
                 + " | ".join("" for _ in Es) + " |")

        def row(lab, f):
            L.append(f"| {lab} | " + " | ".join(f(by[(E, kappa)]) if (E, kappa) in by else "--"
                                                for E in Es) + " |")
        row(f"rbar_f^{kappa} (ann. pp)", lambda r: _d3(_ann(r["rbar"])))
        row(f"c-bar^{kappa} (ann. pp)", lambda r: f"{_d3(_ann(r['cbar']))} "
            f"({_d3(_ann_sd(r['cbar'], r['cbar_se']))})" if r.get("cbar_se") is not None
            else _d3(_ann(r["cbar"])))
        row("95% CI", lambda r: f"[{_d3(_ann(r['ci_lo']))}, {_d3(_ann(r['ci_hi']))}]"
            if r.get("ci_lo") is not None else "--")
        row(FRAC_BIND_LABEL_MD, lambda r: _d3(r["frac_bind"]))
    fbs = [r["frac_bind"] for r in rows if r.get("frac_bind") is not None]
    lead = ("c-bar^kappa = omega^kappa + rbar_f^kappa * zeta^kappa, the marginal cost of deposits "
            "at the type's forward rate rbar_f^kappa (first row of each panel), where "
            + _rbar_def_md(dead=bool(_dead_by_E(rows))) + ". ")
    if identified and _unidentified(rows):
        lead += ("c-bar is identified in every block. Where omega and zeta are separately "
                 "identified as well (launch-quarter variation in dpsi_4/dpsi_2, see the by-start "
                 "ridge table), it restates them at one rate; where they are not ("
                 + _md_unident_ref(rows) + "), it is the interpretable cost object. ")
    elif identified:
        lead += ("c-bar is identified in every block, and so are omega and zeta separately "
                 "(launch-quarter variation in dpsi_4/dpsi_2, see the by-start ridge table); it "
                 "restates them at one rate. ")
    else:
        lead += ("It is the only cost object this design identifies; omega and zeta separately "
                 "are not (see the ridge table). ")
    L += ["",
          lead +
          "SE is a "
          "firm-block bootstrap SD; the CI is the subsampling sqrt(n)-rate quantile interval "
          "(draws and seeds at the end of this note, from the solve's record). "
          + _ci_rbar_note(rows, md=True)
          + "The last row is the share of the "
          "firm × launch-quarter × deviation revealed-preference inequalities that FAIL at "
          "theta-hat — "
          "deviations the model says would have raised the firm's value, i.e. price moves the "
          "bank should have made and did not. A genuine best response implies a share near 0; "
          f"here it is [{min(fbs):.3f}, {max(fbs):.3f}], and 1/2 is the mechanical value of a "
          "symmetric ± grid around a policy with no interior turning point (exactly one of each "
          "± pair fails for ANY theta)."
          # Wherever the share sits at its mechanical value -- always where the split is
          # unidentified, and in the identified regime when every share is within MECH_BAND of
          # 1/2 (the 2026-09-22 run once its dead firm-quarters are out).
          + (" It carries no information about fit, so these magnitudes are diagnostics rather "
             "than estimates." if (not identified or _mechanical(rows)) else "")
          + (" " + _dead_ref(rows, md=True).rstrip() if _dead_by_E(rows) else "")
          + (" " + disc if disc else "")]
    return "\n".join(L)


def build_cbar_design(rows_single, rows_multi, n_starts=None, disc=""):
    r"""c-bar and its criterion health under the SINGLE-CURVE design: one column per routine, one
    panel per firm type, from the cost_params the single-curve solve wrote.

    The multi-start columns this table used to carry beside them repeated tab_bbl_cbar cell for
    cell, so they were dropped (user, 2026-09-29): the multi-start numbers are read in
    tab_bbl_cbar. `rows_multi` and `n_starts` still arrive, from the same --compare-single-curve
    call, and name the multi-start design the note points to.
    """
    sg = {(r["E"], r["block"]): r for r in rows_single}
    # One column per routine of either design: a routine the single-curve run lacks is dashed,
    # not silently dropped.
    have = set(sg) | {(r["E"], r["block"]) for r in rows_multi}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in have for k, _ in BLOCKS)]
    ncol = 1 + len(Es)
    col_fmt = (r">{\raggedright\arraybackslash}p{5.0cm} "
               rf"*{{{len(Es)}}}{{>{{\centering\arraybackslash}}X}}")
    header = " & " + " & ".join(rc.est_ref(E) for E in Es)
    body = []
    for pi, (kappa, _lbl) in enumerate(BLOCKS):
        if not any((E, kappa) in have for E in Es):
            continue
        title = ("Brick-and-Mortar (B) Firms" if kappa == "B" else "Digital (D) Firms")
        if pi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Panel {'AB'[pi]}: {title}}}}} \\")
        body.append(r"\addlinespace[0.3ex]")

        def cells(f, dash="---"):
            out = []
            for E in Es:
                r = sg.get((E, kappa))
                out.append(f(r) if r else dash)
            return " & ".join(out)

        body.append(_rbar_label(kappa) + " & "
                    + cells(lambda r: _cell_rbar(r, "T11")) + r" \\")
        body.append(r"\addlinespace[0.3ex]")
        body.append(rf"$\hat{{\bar c}}^{{\mathrm{{{kappa}}}}}$ & "
                    + cells(lambda r: _cell_cbar(r, "T11")) + r" \\")
        body.append(" & " + cells(lambda r: _cell_cbar_sd(r, "T11"), dash="") + r" \\")
        body.append(r"\quad 95\% CI & " + cells(lambda r: _cell_cbar_ci(r, "T11")) + r" \\")
        body.append(r"\addlinespace[0.4ex]")
        body.append(FRAC_BIND_LABEL + " & " + cells(lambda r: _cell_share(r, "T11")) + r" \\")
    starts = f"{n_starts} launch quarters" if n_starts else "several launch quarters"
    # Notes held to a few lines (V_Main style guide): what the single-curve design is, that rows
    # and units follow tab_bbl_cbar (where the multi-start design is reported), and the mechanical
    # share; the draws and seeds and the discount sentence of the single-curve solve come in `disc`.
    foot = (
        r"\textit{Notes:} Single-curve design: every simulated path is priced off one Focus "
        r"forward curve, so only $\bar c^\kappa$ is identified, and $\bar r^{f,\kappa}$ runs over "
        r"firm~$\times$~deviation rows. Rows and units as in Table~\ref{tab:bbl_cbar}, which "
        rf"reports the multi-start design ({starts}, one Focus curve each). "
        + (r"All shares lie within $0.050$ of the mechanical $1/2$. "
           if _mechanical(list(rows_single)) else "")
    ).rstrip()
    return _wrap(body, col_fmt, r"BBL Marginal Cost $\bar c^\kappa$: Single-Curve Design",
                 "tab:bbl_cbar_design", header, _with(foot, disc), ncol)


def md_cbar_design(rows_single, rows_multi, disc=""):
    # The markdown twin of build_cbar_design: the single-curve design only (the multi-start
    # columns repeated the c-bar table). `rows_multi` is kept in the signature for the caller.
    sg = {(r["E"], r["block"]): r for r in rows_single}
    have = set(sg) | {(r["E"], r["block"]) for r in rows_multi}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in have for k, _ in BLOCKS)]
    L = ["| | " + " | ".join(f"E{E} single curve" for E in Es) + " |",
         "|---|" + "---:|" * len(Es)]
    for kappa, _ in BLOCKS:
        L.append(f"| **{'Brick-and-Mortar (B)' if kappa == 'B' else 'Digital (D)'}** | "
                 + " | ".join("" for _ in range(len(Es))) + " |")

        def row(lab, f):
            out = []
            for E in Es:
                r = sg.get((E, kappa))
                out.append(f(r) if r else "--")
            L.append(f"| {lab} | " + " | ".join(out) + " |")
        row(f"rbar_f^{kappa} (ann. pp)", lambda r: _d3(_ann(r["rbar"])))
        row(f"c-bar^{kappa} (SE), ann. pp", lambda r: f"{_d3(_ann(r['cbar']))} "
            f"({_d3(_ann_sd(r['cbar'], r['cbar_se']))})" if r.get("cbar_se") is not None
            else _d3(_ann(r["cbar"])))
        row("95% CI", lambda r: f"[{_d3(_ann(r['ci_lo']))}, {_d3(_ann(r['ci_hi']))}]"
            if r.get("ci_lo") is not None else "--")
        row("violated share", lambda r: _d3(r["frac_bind"])
            if r.get("frac_bind") is not None else "--")
    L += ["", "Single-curve design: one Focus forward curve for every path, so c-bar is the only "
              "identified cost object. c-bar^kappa = omega^kappa + rbar_f^kappa * zeta^kappa, where "
              + _rbar_def_md(SINGLE_ROWS_MD, dead=False)
              + ". Rows and units as in the c-bar table, which reports the multi-start design. "
              + "The violated share's mechanical value is 1/2"
              + ("; every share in the table lies within 0.05 of it, so the row carries no "
                 "information about fit" if _mechanical(list(rows_single)) else "") + ". "
              + (disc if disc else "")]
    L[-1] = L[-1].rstrip()
    return "\n".join(L)


def _ci_cell(ci):
    r"""One subsampling interval as a table cell, or '---' when the solve did not invert the
    criterion for that parameter. An EMPTY confidence set is printed as such: it is a finding
    about the criterion (nothing was ever within the critical value), not a missing number, and
    printing it as a dash would hide the one case a reader most needs to see. A window-truncated
    endpoint is flagged for the same reason -- the profile grid ran out before the criterion did,
    so that end is not a set boundary."""
    if not ci:
        return "---"
    if ci.get("empty"):
        return r"{\scriptsize \textit{empty}}"
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if lo is None or hi is None:
        return "---"
    mark = r"$^{\dagger}$" if (ci.get("truncated_lo") or ci.get("truncated_hi")) else ""
    return rf"{{\scriptsize $[{_m3(lo)},{_m3(hi)}]${mark}}}"


def build_identified_panels(rows, identified=False, disc="", rs=None):
    r"""Estimates table in the paper's own layout: parameters down the rows, estimation routines
    across the columns as \ref{estimation:*} (rendered (III)-(VI)), and one panel per firm type.

    `rs` is _resampling_parts of the solves behind the table: their draws and seeds are named
    where the notes name the intervals and the SDs, so this table carries no separate resampling
    sentence (user, 2026-09-29).

    This replaces a routine x type row grid, which needed eight columns and had no room for
    gamma at all. Five columns fit the text block comfortably, standard errors sit under their
    estimates as in polfunc_k4.tex, and the type superscript moves out of every cell into the
    panel title -- V_Main's notation is omega^{\mathrm{B}}, zeta^{\mathrm{B}},
    (\boldsymbol{\gamma}^{\mathrm{B}})', and \bar c^\kappa for their combination.

    `identified` decides what this table IS. When the solve separated the split, omega and zeta
    are the estimand and each carries the subsampling interval from the inverted criterion; the
    stars, which need a normal reference the criterion does not provide, come off. When it did
    not, the table stays what it has been -- a completeness report whose notes say so.
    """
    by = {(r["E"], r["block"]): r for r in rows}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in by for k, _ in BLOCKS)]
    ncol = 1 + len(Es)
    col_fmt = (r">{\raggedright\arraybackslash}p{5.0cm} "
               rf"*{{{len(Es)}}}{{>{{\centering\arraybackslash}}X}}")
    header = " & " + " & ".join(rc.est_ref(E) for E in Es)

    def line(label, get, fmt="{:.3f}", bold=False, se_get=None, se_fmt="{:.3f}", stars=False,
             flag=False):
        """One estimate row, plus a parenthesized SE row underneath when se_get is given.
        `flag` appends the double dagger when this block's solve said the omega/zeta split is
        not identified -- the mark goes on the estimate, where the eye lands."""
        cells = []
        for E in Es:
            r = by.get((E, kappa))
            v = get(r) if r else None
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("---")
            else:
                # Counts ({:,.0f}) as integers; every other value through _m3 (three decimals,
                # braced separators, no -0.000, nonzero zeros listed).
                s = (fmt.format(v).replace(",", "{,}") if fmt.endswith(".0f}")
                     else _m3(v, f"T7 {rc.est_ref(E)} {kappa} {_plain(label)}"))
                mk = r"^{\ddagger}" if flag and r.get("identified") is False else ""
                if bold:
                    cells.append(r"\textbf{" + s + "}")
                elif stars:
                    cells.append(f"${s}^{{{_stars(v, se_get(r) if se_get else None)}}}$")
                else:
                    cells.append(f"${s}{mk}$")
        # longtable's \\* forbids a page break after the row, so an estimate and the SE row
        # under it always land on the same page.
        out = [label + " & " + " & ".join(cells) + (r" \\*" if se_get is not None else r" \\")]
        if se_get is not None:
            ses = []
            for E in Es:
                r = by.get((E, kappa))
                sv = se_get(r) if r else None
                ses.append(f"$({_m3(sv, f'T7 {rc.est_ref(E)} {kappa} {_plain(label)} SD')})$"
                           if sv is not None and np.isfinite(sv) else "")
            out.append(" & " + " & ".join(ses) + r" \\")
        return out

    def ci_line(label, key):
        """A subsampling-interval row under an estimate. Only used in the identified branch,
        where the interval and not the SE is the uncertainty statement."""
        cells = []
        for E in Es:
            r = by.get((E, kappa))
            cells.append(_ci_cell(r.get(key)) if r else "---")
        return [label + " & " + " & ".join(cells) + r" \\"]

    body = []
    has_cond = any(r.get("cond_pooled") is not None for r in rows)
    # In the identified branch the symbols the cells carry (the dagger on a profile-window
    # endpoint, the double dagger on a block whose split failed the gate) are defined in the
    # Notes, like every other symbol of these tables, and the body opens on Panel A.
    if not identified:
        # The kinked-criterion caveat, in the table itself and not only mid-footnote: the reader
        # meets it before scanning any stars. The full statement stays in the Notes.
        body.append(
            rf"\multicolumn{{{ncol}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}{{\scriptsize\itshape Significance stars are shown by "
            rf"convention only --- the kinked criterion admits no normal reference (see Notes).}} \\")
        body.append(r"\addlinespace[0.4ex]")
    for pi, (kappa, _lbl) in enumerate(BLOCKS):
        if not any((E, kappa) in by for E in Es):
            continue
        title = ("Brick-and-Mortar (B) Firms" if kappa == "B" else "Digital (D) Firms")
        if pi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Panel {'AB'[pi]}: {title}}}}} \\")
        body.append(r"\addlinespace[0.3ex]")
        body += line(rf"$\hat\omega^{{\mathrm{{{kappa}}}}}$",
                     lambda r: r["omega"], se_get=lambda r: r.get("omega_se"),
                     stars=not identified, flag=identified)
        if identified:
            body += ci_line(r"\quad 95\% CI", "ci_omega")
        # Three decimals like every other estimate and SD in the table.
        body += line(rf"$\hat\zeta^{{\mathrm{{{kappa}}}}}$",
                     lambda r: r["zeta"],
                     se_get=lambda r: r.get("zeta_se"),
                     stars=not identified, flag=identified)
        if identified:
            body += ci_line(r"\quad 95\% CI", "ci_zeta")
        body.append(r"\addlinespace[0.3ex]")
        # The gamma block is headed by the bare symbol, flush left like the other parameters;
        # the shifters it loads on are indented beneath it.
        body.append(rf"$\hat{{\boldsymbol{{\gamma}}}}^{{\mathrm{{{kappa}}}}}$"
                    + " & " * len(Es) + r" \\")
        for key, lab in Z_LABELS:
            body += line(r"\hspace{1em}" + lab,
                         (lambda k: lambda r: _zscaled((r["gamma"] or {}).get(k), k))(key),
                         se_get=(lambda k: lambda r: _zscaled((r["gamma_se"] or {}).get(k), k))(key),
                         stars=not identified)
        # Regression information, separated from the parameters as in the sleepiness tables.
        # (rbar^f is not a parameter of eq:8 — it lives with c-bar, which it defines.)
        body.append(r"\midrule")
        body += line(r"Firms", lambda r: r["n_firms"], fmt="{:,.0f}")
        body += line(r"Inequalities $n$", lambda r: r["n"], fmt="{:,.0f}")
        # The statistic the omega/zeta split is judged on, per block: the pooled condition index
        # of [dpsi2 dpsi4] the solve's gate reads. Kept here, beside the double dagger it decides,
        # because the ridge tables pool the two firm types and so never show it.
        if has_cond:
            body += line(r"Condition index", lambda r: r.get("cond_pooled"), flag=identified)

    # Notes held to a few lines (V_Main style guide, 2026-09-28): sample and units, what the
    # brackets and parentheses hold (the profile covers omega and zeta only), the symbols the cells
    # carry -- a dagger or "empty" only when a cell shows one -- the condition-index row with its
    # double dagger, the dead firm-quarters behind n, and the discount sentence.
    cis = [r.get(k) or {} for r in rows for k in ("ci_omega", "ci_zeta")]
    has_dagger = any(c.get("truncated_lo") or c.get("truncated_hi") for c in cis)
    has_empty = any(c.get("empty") for c in cis)
    thr = next((r.get("cond_max") for r in rows if r.get("cond_max")), 30.0)
    # The lag is in every shifter's row label; the units the labels do not carry are stated here.
    # They are the units the cost block's Z enters in: the demand-prep parquets that cf_psi_basis.jl
    # reads carry the cost ratios as fractions of assets and the Basel index x100 (median 16.3).
    # Trimmed with the user (2026-09-29): the coefficients are per quarter because a linear
    # coefficient cannot be compounded, which the text says; the note keeps the units only.
    common = (
        r"\textit{Notes:} Coefficients in per-quarter units, the Basel one in basis points; cost "
        r"ratios in fractions of assets, Basel index in percentage points. "
    )
    rs = rs or {}
    sub = f" ({rs['sub']})" if rs.get("sub") else ""
    boot = f" ({rs['boot']})" if rs.get("boot") else ""
    cond = ""
    if has_cond:
        cond = ((rf"$\ddagger$: condition index of $[\Delta\psi_2\;\Delta\psi_4]$ (last row) "
                 rf"above {thr:.0f}, so $\omega^\kappa$ and $\zeta^\kappa$ are not separately "
                 r"identified. ")
                if identified and _unidentified(rows) else
                (r"Last row: Belsley--Kuh--Welsch condition index of "
                 r"$[\Delta\psi_2\;\Delta\psi_4]$"
                 + (rf", below {thr:.0f} in every block. " if identified else ". ")))
    if identified:
        # One sentence for both uncertainty measures, with the draws and seeds of each (user,
        # 2026-09-29); it restates the user's 2026-09-28 wording: the criterion inversion profiles
        # (omega, zeta) only, and the loadings gamma carry descriptive firm-block bootstrap SDs.
        foot = (
            common +
            r"Brackets: 95\% intervals inverting \eqref{eq:16} at a subsampled critical value"
            + sub + r", for $(\omega^\kappa, \zeta^\kappa)$ only"
            + (r"; $\dagger$: profile-window endpoint" if has_dagger else "")
            + (r"; \textit{empty}: never within the critical value" if has_empty else "")
            + r"; parentheses: firm-block bootstrap standard deviations" + boot
            + r", descriptive for the loadings $\gamma^\kappa$. " + cond + _dead_note(rows)
        )
    else:
        foot = (
            common +
            r"Parentheses: firm-block bootstrap SDs" + boot + r"; stars use their normal "
            r"approximation and are "
            r"shown by convention only, the criterion being kinked "
            r"(Section~\ref{sec:empirical:cost}): *** $p<0.01$, ** $p<0.05$, * $p<0.1$. The "
            r"parameters are not separately identified; the estimand is $\bar c^\kappa$ "
            r"(Table~\ref{tab:bbl_cbar}). " + cond + _dead_note(rows)
        )
    foot = foot.rstrip()
    return _wrap(body, col_fmt,
                 r"BBL Deposit-Servicing Cost Parameters, by Firm Type",
                 "tab:bbl_cost_identified", header, _with(foot, disc), ncol)


def _md_ci(ci):
    if not ci:
        return "--"
    if ci.get("empty"):
        return "*empty*"
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if lo is None or hi is None:
        return "--"
    return (f"[{_d3(lo)}, {_d3(hi)}]"
            + ("†" if (ci.get("truncated_lo") or ci.get("truncated_hi")) else ""))


def md_identified_panels(rows, identified=False, disc=""):
    by = {(r["E"], r["block"]): r for r in rows}
    Es = [E for E in ROUTINE_ORDER if any((E, k) in by for k, _ in BLOCKS)]
    L = ["| | " + " | ".join(f"E{E}" for E in Es) + " |",
         "|---|" + "---:|" * len(Es)]

    def row(lab, f):
        L.append(f"| {lab} | " + " | ".join(f(by[(E, kappa)]) if (E, kappa) in by else "--"
                                            for E in Es) + " |")
    for kappa, _ in BLOCKS:
        if not any((E, kappa) in by for E in Es):
            continue
        L.append(f"| **Panel {'A' if kappa=='B' else 'B'}: "
                 f"{'Brick-and-Mortar (B)' if kappa=='B' else 'Digital (D)'} Firms** | "
                 + " | ".join("" for _ in Es) + " |")
        row(f"omega^{kappa}", lambda r: f"{_d3(r['omega'])}"
                                        f"{'' if identified else _stars(r['omega'], r['omega_se'])}"
                                        f"{'‡' if identified and r.get('identified') is False else ''} "
                                        f"({_d3(r['omega_se'])})")
        if identified:
            row("  95% CI", lambda r: _md_ci(r.get("ci_omega")))
        row(f"zeta^{kappa}", lambda r: f"{_d3(r['zeta'])}"
                                       f"{'' if identified else _stars(r['zeta'], r['zeta_se'])}"
                                       f"{'‡' if identified and r.get('identified') is False else ''} "
                                       f"({_d3(r['zeta_se'])})")
        if identified:
            row("  95% CI", lambda r: _md_ci(r.get("ci_zeta")))
        row(f"**gamma^{kappa}**", lambda r: "")
        for key, lab in Z_LABELS:
            row("  " + lab.replace("($t-1$)", "(t-1)"),
                (lambda k: lambda r: (
                    f"{_d3(_zscaled((r['gamma'] or {}).get(k, float('nan')), k))}"
                    f"{'' if identified else _stars((r['gamma'] or {}).get(k), (r['gamma_se'] or {}).get(k))} "
                    f"({_d3(_zscaled((r['gamma_se'] or {}).get(k, float('nan')), k))})"))(key))
        row("*Firms*", lambda r: f"{r['n_firms']:,}")
        row("*Inequalities n*", lambda r: f"{r['n']:,}")
        if any(r.get("cond_pooled") is not None for r in rows):
            row("*Condition index*", lambda r: (
                "--" if r.get("cond_pooled") is None else
                f"{_d3(r['cond_pooled'])}{'‡' if identified and r.get('identified') is False else ''}"))
    units = ("Coefficients in native per-quarter units (a linear coefficient cannot be "
             "compounded); in psi_3 the personnel, administrative and tax cost ratios are "
             "fractions of total assets and the Basel index is in percentage points, all lagged "
             "one quarter, and each gamma coefficient is per unit of its shifter; the Basel one "
             "is shown in basis points (bp of quarterly marginal cost per pp of the index). ")
    if identified:
        L += ["",
              "Columns are estimation routines. " + units + _dead_note(rows, md=True)
              + "**omega and zeta are separately identified "
              "here**: each launch quarter is simulated from the Focus forward curve published at "
              "that quarter, so dpsi_4/dpsi_2 varies across quarters and dpsi_2 and dpsi_4 fall "
              "below the collinearity threshold "
              "(see the by-start ridge table); *Condition index* is the pooled Belsley-Kuh-Welsch "
              "index of [dpsi_2 dpsi_4] in the block, judged against 30. The bracketed 95% CI "
              "inverts the eq:16 criterion "
              "at a subsampled critical value — the appropriate route for a kinked, potentially "
              "set-identified criterion; the inversion profiled omega and zeta only, so gamma "
              "carries no interval. † marks a profile-window endpoint (a limit "
              "of the search, not a set boundary) and *empty* means the criterion was never "
              "within the critical value. Parenthesized figures are firm-block bootstrap SDs, "
              "descriptive only. c-bar^kappa restates the pair at the type's forward rate "
              "rbar_f^kappa, defined with the c-bar table." + _md_unident(rows)
              + " " + _md_n_ineq(rows) + (" " + disc if disc else "")]
    else:
        L += ["",
              "Columns are estimation routines. " + units + _dead_note(rows, md=True)
              + "SEs in parentheses are firm-block bootstrap SDs "
              "(draws and seed at the end of this note). **None of these parameters is separately "
              "identified** — omega, zeta and "
              "gamma slide freely along the ridge (one block returns omega>0 with zeta<0 while c-bar "
              "barely moves); the estimand is c-bar, reported separately. Stars use the normal "
              "approximation to the bootstrap SD and are shown by convention only: the criterion is "
              "kinked and potentially set-identified, so they should not be read as tests. "
              "*** p<0.01, ** p<0.05, * p<0.1. " + _md_n_ineq(rows)
              + (" " + disc if disc else "")]
    return "\n".join(L)


def md_ridge(ridge, disc="", by_start=None):
    # The markdown twin of build_ridge, trimmed the same way (user, 2026-09-29). Short headers
    # on purpose: pandoc/xelatex renders this at 11pt in a portrait page. Symbols are defined in
    # the line below the table.
    within = {E: _within_cond_min((by_start or {}).get(E)) for E in ridge}
    has_within = any(v[0] is not None for v in within.values())
    L = ["| Routine | n | Corr | Cond. | " + ("Cond. (min. quarter) | " if has_within else "")
         + "Median ratio | IQR% |",
         "|---|---:|---:|---:|" + ("---:|" if has_within else "") + "---:|---:|"]
    for E in sorted(ridge):
        s = ridge[E]
        wq = (f"{_md_cond(within[E][0])} ({within[E][1]}) | " if within[E][0] is not None
              else "-- | ") if has_within else ""
        L.append(f"| E{E} | {s['n']:,} | {_d3(s['corr'])} | {_d3(s['cond'])} | {wq}"
                 f"{_d3(_ann(s['ratio_median']))} | {_d3(_iqr_ann(s))} |")
    L += ["",
          _ridge_rows_note(ridge, md=True)
          + "*Corr* = corr(dpsi_2, dpsi_4); *Cond.* = Belsley-Kuh-Welsch (1980) condition index of "
          "[dpsi_2 dpsi_4], i.e. with the columns scaled to unit length, so it measures "
          "collinearity alone and not the columns' units; >30 is the usual threshold"
          + ("; *Cond. (min. quarter)* = its lowest value within a single launch quarter, "
             "with that quarter" if has_within else "")
          + ". *Median ratio* = median dpsi_4/dpsi_2 over the n rows, in compounded annual pp "
          "((1+x)^4-1)*100 (within one firm type it is rbar_f^kappa of the c-bar table); "
          "*IQR%* = interquartile range as a percent of the median. No resampling is behind "
          "these numbers." + (" " + disc if disc else "")]
    return "\n".join(L)


# ── machine-readable numbers: bbl_table_numbers{psi_tag}.json ────────────────────
# make_paper_numbers.py reads every BBL number the paper's prose quotes from this file, never from
# the .tex. Each pass records what it rendered, from the objects it rendered it from (the block
# rows, the ridge and per-quarter records, the up/down split, the beta/T provenance), and merges
# only its own sections into the one file of the run's psi tag, in the step folder:
#   default pass   cost_tables (tab_bbl_cost_identified, tab_bbl_cbar), tab_bbl_cbar_design
#   --from-psi     tab_bbl_ridge_diagnostic, tab_bbl_ridge_by_start, tab_bbl_violated_by_sign
# Values are unrounded; a rate level also carries its compounded annual value (the *_ann fields,
# _ann/_ann_sd), the scale the tables print it on. Every section carries the (omega, zeta) of the
# blocks it describes (`theta`), so a reader can tell whether two sections describe one solve.
def _numbers_path(tag: str) -> pathlib.Path:
    return STEP_DIR / f"bbl_table_numbers{tag or ''}.json"


def _numbers_tag(tags) -> str:
    """The psi tag the numbers file is named after: the one tag every routine carries, or
    '_mixed' (said on stdout) when the routines of one pass come from different runs."""
    ts = sorted(set(tags))
    if len(ts) == 1:
        return ts[0]
    print(f"  !!!! [numbers] the routines carry different psi tags {ts}; written to "
          f"{_numbers_path('_mixed').name} !!!!")
    return "_mixed"


def _jsonable(x):
    """x with numpy scalars as Python numbers and every non-finite float as None."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _ann_or_none(x):
    return None if x is None or not np.isfinite(float(x)) else _ann(x)


def _ci_numbers(ci):
    """One profile interval of tab_bbl_cost_identified, the fields _ci_cell renders."""
    if not ci:
        return None
    return {k: ci.get(k) for k in ("ci_lo", "ci_hi", "empty", "truncated_lo", "truncated_hi")}


def _row_numbers(r: dict) -> dict:
    """One block row of collect_json() as the cost tables print it (T7, T9, T11)."""
    se = r.get("cbar_se")
    return dict(
        E=r["E"], block=r["block"], identified=r.get("identified"),
        cond_pooled=r.get("cond_pooled"), cond_max=r.get("cond_max"),
        corr_pooled=r.get("corr_pooled"),
        omega=r["omega"], omega_se=r.get("omega_se"), ci_omega=_ci_numbers(r.get("ci_omega")),
        zeta=r["zeta"], zeta_se=r.get("zeta_se"), ci_zeta=_ci_numbers(r.get("ci_zeta")),
        gamma=r.get("gamma"), gamma_se=r.get("gamma_se"), gamma_display_scale=Z_SCALE,
        n=r.get("n"), n_firms=r.get("n_firms"), n_ratio=r.get("n_ratio"),
        null_mode=r.get("null_mode"), n_null_fq=r.get("n_null_fq"),
        n_null_rows=r.get("n_null_rows"),
        rbar=r["rbar"], rbar_ann=_ann(r["rbar"]), cbar=r["cbar"], cbar_ann=_ann(r["cbar"]),
        cbar_se=se, cbar_sd_ann=_ann_sd(r["cbar"], se) if se is not None else None,
        ci_lo=r.get("ci_lo"), ci_hi=r.get("ci_hi"),
        ci_lo_ann=_ann_or_none(r.get("ci_lo")), ci_hi_ann=_ann_or_none(r.get("ci_hi")),
        ci_rbar=r.get("ci_rbar"), ci_rbar_ann=_ann_or_none(r.get("ci_rbar")),
        frac_bind=r.get("frac_bind"))


def _cost_file_names(Es, tag) -> dict:
    """{'E<k>': the cost_params file _cost_path reads for each routine}, without its stdout note."""
    return {f"E{E}": (c[0].name if (c := _cost_candidates(E, tag)) else None) for E in sorted(Es)}


def _blocks_numbers(rows) -> dict:
    return {f"E{r['E']}_{r['block']}": _row_numbers(r) for r in rows}


def _theta_of(rows) -> dict:
    return {f"E{r['E']}_{r['block']}": [float(r["omega"]), float(r["zeta"])] for r in rows}


def _ridge_numbers(s: dict) -> dict:
    """One ridge_stats record: a tab_bbl_ridge_diagnostic row or a tab_bbl_ridge_by_start row."""
    out = {k: s[k] for k in ("n", "n_all", "n_nonfinite", "n_d2_zero", "n_firms", "n_shock",
                             "corr", "cond", "cond_raw", "ratio_median", "ratio_q25",
                             "ratio_q75", "iqr_pct", "ratio_mean", "cv", "cv_top10_ss_pct",
                             "one_minus_r2", "across_delta_pct", "n_starts", "n_loaded",
                             "n_dead_rows", "n_dead_fq", "n_ineq", "null_mode", "n_shards")
           if k in s}
    out.update(ratio_median_ann=_ann(s["ratio_median"]), ratio_mean_ann=_ann(s["ratio_mean"]),
               iqr_pct_ann=_iqr_ann(s))
    return out


def _prov_numbers(prov: dict, tags: dict) -> dict:
    """{'E<k>': tag, beta, T, beta_annual, coverage, source} for the run behind each routine, as
    the table's discount sentence states it (None where the run recorded no beta/T)."""
    out = {}
    for E in sorted(set(prov) | set(tags)):
        p = prov.get(E)
        rec = dict(tag=tags.get(E), beta=None, T=None, source=None)
        if p is not None:
            rec.update(beta=float(p["beta"]), T=int(p["T"]), source=p.get("source"),
                       beta_annual=_disc.annual(p["beta"]),
                       coverage=_disc.coverage(p["beta"], p["T"]))
        out[f"E{E}"] = rec
    return out


def _write_numbers(tag: str, sections: dict, source: str) -> None:
    """Merge this pass's sections into bbl_table_numbers{tag}.json; the other pass's sections
    already in the file are kept. Written through a temporary file and a rename."""
    p = _numbers_path(tag)
    doc = {}
    if p.is_file():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"  !!!! [numbers] {p.name} unreadable; rewritten with this pass's sections "
                  f"only !!!!")
            doc = {}
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    doc.update(kind="bbl_table_numbers", psi_tag=tag, generator="make_bbl_cost_tables.py")
    secs = doc.setdefault("sections", {})
    for name, sec in sections.items():
        secs[name] = dict(sec, written=stamp, source=source)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_jsonable(doc), indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)
    print(f"  wrote {p.name} [{', '.join(sections)}] -> {p.parent}")


def main_from_json(compare_single=False, single_tag=""):
    """Build both tables from cost_params alone (default). Vintage-safe by construction: every
    column, ridge diagnostics included, comes from the solve's own record of the psi it used."""
    ZERO_PRINTS.clear()
    cost = load_cost_only()
    if not cost:
        raise SystemExit(f"no cost_params found in {COST_DIR}")
    # Dead firm-quarters (DEAD FIRM-QUARTERS in the module docstring). A solve that applied the
    # rule is read as written; a multi-start cost_params that predates it is restated from the
    # record the --from-psi pass writes, and left as the solve wrote it (said loudly) without one.
    for E in sorted(cost):
        mode = _null_fq_mode(cost[E])
        if mode == "table_mask":
            rec = _read_null_fq_record(E, cost[E])
            if rec:
                cost[E] = _overlay_null_fq(cost[E], rec)
        elif mode in ("solve_drop", "solve_keep"):
            c = _solve_dead_counts(cost[E])
            print(f"  [dead fq] E{E}: {mode}; the solve recorded "
                  + ", ".join(f"{k} {v.get('n_null_fq')} fq / {v.get('n_null_rows')} rows"
                              for k, v in c.items()))
            for kappa in ((((cost[E].get("run") or {}).get("null_fq") or {})
                           .get("types_emptied")) or []):
                print(f"  !!!! [dead fq] E{E}: every {kappa} firm-quarter is dead, so the solve "
                      f"wrote no {kappa} block -- its cells are dashed !!!!")
    rb, rows = collect_json(cost)
    if not rows:
        raise SystemExit("cost_params contained no B/D blocks.")
    for s in rb:
        print(f"  E{s['E']}-{s['block']}: n={s['n']:,} firms={s['n_firms']} "
              f"corr={s['corr']:.8f} cond={s['cond']:,.0f} rbar={s['rbar']:.5f} "
              f"1-R2={s['one_minus_r2']:.2e} frac_bind={s['frac_bind']:.3f}")
    fbs = [s["frac_bind"] for s in rb if s["frac_bind"] is not None]
    if fbs and min(fbs) >= 0.45 and max(fbs) <= 0.55:
        print(f"\n  !!!! every block has frac_bind in [{min(fbs):.3f}, {max(fbs):.3f}] — the "
              "mechanical value.\n       The magnitudes below are diagnostics, not estimates.\n")
    ident = identified_split(cost)
    for r in rows:
        print(f"    E{r['E']}-{r['block']}: identified_split={r.get('identified')}  "
              f"pooled cond={_fmt_cond(r.get('cond_pooled'))}  corr={r.get('corr_pooled')}")
    print(f"  identified_split = {ident}  -> "
          + ("omega/zeta are the estimand; c_bar restates them at one rate." if ident
             else "c_bar is the estimand; the omega/zeta split is reported for completeness."))
    # beta and T as the run behind each routine recorded them (never the registry's).
    prov = {E: _run_discount(E, _cost_tag(cost[E], PSI_TAG), cost[E]) for E in cost}
    Es = [E for E in ROUTINE_ORDER if any(r["E"] == E for r in rows)]
    d7 = _discount_report("tab_bbl_cost_identified", _routine_entries(Es, prov))
    d9 = _discount_report("tab_bbl_cbar", _routine_entries(Es, prov))
    # Only the two cost tables come from cost_params. The ridge tables are per-routine over the
    # pooled design and are built from the psi by --from-psi; they are deliberately NOT
    # written here, so a cost-side rebuild cannot restyle or overwrite them.
    # The draws and seeds of the solve's bootstrap and subsampling, ahead of the discount sentence;
    # Table 7's TeX note names them inside its brackets/parentheses sentence instead.
    rs = (_resampling_note(cost.values()), _resampling_note(cost.values(), md=True))
    tex = {"tab_bbl_cbar.tex": build_cbar_panels(rows, ident, disc=_notes_join(rs[0], d9[0])),
           "tab_bbl_cost_identified.tex": build_identified_panels(
               rows, ident, disc=d7[0], rs=_resampling_parts(cost.values()))}
    md = {"tab_bbl_cbar.md": md_cbar_panels(rows, ident, disc=_notes_join(rs[1], d9[1])),
          "tab_bbl_cost_identified.md": md_identified_panels(
              rows, ident, disc=_notes_join(rs[1], d7[1]))}
    if compare_single:
        # The single-curve side is pinned to the files of `single_tag` (--single-curve-tag; the
        # UNTAGGED files when it is empty), the multi-start side to this run's tag; comparing a
        # multi-start run with itself, or with whatever vintage a loose glob returns first,
        # would produce a table that looks right and says nothing.
        if not (PSI_TAG or "").startswith("_ms"):
            raise SystemExit("--compare-single-curve needs the multi-start run pinned, e.g. "
                             "--psi-tag _ms1")
        if single_tag == PSI_TAG:
            raise SystemExit(f"--single-curve-tag {single_tag} is the multi-start run itself")
        sc_what = f"'{single_tag}'-tagged" if single_tag else "untagged"
        single, single_files = {}, {}
        for E in cost:
            p = _cost_path(E, single_tag)
            if p is None:
                print(f"  [design] E{E}: no {sc_what} single-curve cost_params -- its column is "
                      f"dashed")
                continue
            single[E] = json.loads(p.read_text(encoding="utf-8"))
            single_files[f"E{E}"] = p.name
            print(f"  [design] single curve E{E}: {p.name}")
        if not single:
            raise SystemExit(f"--compare-single-curve: no {sc_what} (single-curve) cost_params "
                             f"found in {COST_DIR}")
        _, rows_single = collect_json(single)
        n_starts = next((c.get("run", {}).get("n_starts") for c in cost.values()
                         if c.get("run", {}).get("n_starts")), None)
        prov_single = {E: _run_discount(E, single_tag, single[E]) for E in single}
        Es11 = [E for E in ROUTINE_ORDER
                if any(r["E"] == E for r in rows_single) or any(r["E"] == E for r in rows)]
        # Both runs' provenance is still reported on stdout; the table shows the single-curve
        # design only, so its discount sentence and its draws and seeds are that solve's own.
        _discount_report("tab_bbl_cbar_design", _design_entries(Es11, prov_single, prov))
        d11 = _discount_report("tab_bbl_cbar_design",
                               _routine_entries([E for E in Es11 if E in prov_single],
                                                prov_single))
        rs11 = (_resampling_note(single.values()), _resampling_note(single.values(), md=True))
        tex["tab_bbl_cbar_design.tex"] = build_cbar_design(
            rows_single, rows, n_starts, disc=_notes_join(rs11[0], d11[0]))
        md["tab_bbl_cbar_design.md"] = md_cbar_design(rows_single, rows,
                                                      disc=_notes_join(rs11[1], d11[1]))
    for name, txt in tex.items():
        _write(name, txt)
    for name, txt in md.items():
        _write(name, txt + "\n")
    # The numbers these tables print, from the rows and provenance they were rendered from.
    tags = {E: _cost_tag(cost[E], PSI_TAG) for E in cost}
    rendered = dict(tag=_numbers_tag(tags.values()), files=_cost_file_names(cost, PSI_TAG),
                    theta=_theta_of(rows), provenance=_prov_numbers(prov, tags),
                    blocks=_blocks_numbers(rows))
    nums = {"cost_tables": dict(rendered, tables=["tab_bbl_cost_identified", "tab_bbl_cbar"],
                                identified_split=ident)}
    if compare_single:
        tags_single = {E: _cost_tag(single[E], single_tag) for E in single}
        nums["tab_bbl_cbar_design"] = dict(
            n_starts=n_starts, multi=rendered,
            single=dict(tag=single_tag, files=single_files, theta=_theta_of(rows_single),
                        provenance=_prov_numbers(prov_single, tags_single),
                        blocks=_blocks_numbers(rows_single)))
    _write_numbers(rendered["tag"], nums, source=f"cost_params in {COST_DIR}")
    # The markdown twins go to stdout as well. Nothing is downloaded from the cluster, so the
    # SLURM log has to BE the deliverable: a table that only exists as a file in the step folder
    # is a table nobody can read without another transfer.
    for name, txt in md.items():
        print(f"\n----- {name} -----\n{txt}")
    print("\n\\input{tab_bbl_cbar.tex}            % tab:bbl_cbar")
    print("\\input{tab_bbl_cost_identified.tex}  % tab:bbl_cost_identified")
    print("\\input{tab_bbl_ridge_diagnostic.tex} % tab:bbl_ridge_diagnostic (--from-psi)")
    print("\\input{tab_bbl_ridge_by_start.tex}   % tab:bbl_ridge_by_start   (--from-psi, _ms)")
    if compare_single:
        print("\\input{tab_bbl_cbar_design.tex}      % tab:bbl_cbar_design      "
              "(--compare-single-curve)")
    print("(needs booktabs + xltabular, both already in V_Main)")
    _report_zero_prints("cost tables (T7, T9, T11)")


def main():
    global COST_DIR, PSI_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi-zip", default=str(PSI_ZIP))
    ap.add_argument("--psi-dir", default=None,
                    help="directory of LOOSE psi_eq/psi_dev parquets to read instead of the zip. "
                         "This is the cluster path: the shards are read where they are written "
                         "(data/output/bbl), before anything is archived.")
    ap.add_argument("--cost-dir", default=None,
                    help="where cost_params_*.json lives. Default: the BBL step folder when "
                         "CF_COST_FWD is exported (cluster), else BBL_OUTPUT/cluster_processed.")
    ap.add_argument("--psi-tag", default=None,
                    help="pin the multi-start tag on every artifact name, e.g. --psi-tag _ms8 "
                         "(pass an empty string for the single-start vintage). Default: match "
                         "either, preferring the largest multi-start run present.")
    ap.add_argument("--from-psi", action="store_true",
                    help="recompute the ridge columns from the psi archive instead of reading "
                         "them out of cost_params. Requires the archive to BE the psi the solve "
                         "consumed; verified by check_psi_matches_solve().")
    ap.add_argument("--compare-single-curve", action="store_true",
                    help="also write tab_bbl_cbar_design: c-bar and its criterion health for the "
                         "single-curve solve (untagged, or --single-curve-tag) next to this "
                         "multi-start run. Needs "
                         "--psi-tag _ms<N> so both sides are pinned.")
    ap.add_argument("--single-curve-tag", default="",
                    help="with --compare-single-curve: the psi tag of the single-curve solve "
                         "whose cost_params fill the Single curve columns, e.g. _sc981. Default: "
                         "the untagged files.")
    ap.add_argument("--per-quarter-cbar", choices=("keep", "drop"), default="drop",
                    help="with --from-psi: keep or drop the per-quarter c-bar column of "
                         "tab_bbl_ridge_by_start (default drop; the user's decision of "
                         "2026-09-28: omega and zeta are not separable within a quarter).")
    ap.add_argument("--allow-vintage-mismatch", action="store_true",
                    help="with --from-psi: emit tables even if the psi archive is not the psi "
                         "the solve used. Only for inspecting a known-mixed pair; not reportable.")
    a = ap.parse_args()

    PSI_TAG = a.psi_tag
    COST_DIR = pathlib.Path(a.cost_dir) if a.cost_dir else _resolve_cost_dir(PSI_TAG)
    print(f"cost_params dir: {COST_DIR}")
    print(f"table dests:     {', '.join(str(d) for d in _dests())}")

    if not a.from_psi:
        return main_from_json(compare_single=a.compare_single_curve,
                              single_tag=a.single_curve_tag)

    zp = pathlib.Path(a.psi_dir) if a.psi_dir else pathlib.Path(a.psi_zip)
    if not zp.exists():
        raise SystemExit(f"psi source not found: {zp}\nPass --psi-zip or --psi-dir.")
    print(f"reading {zp.name}"
          + ("/ (loose shards)" if zp.is_dir() else f" ({zp.stat().st_size/1e6:.1f} MB)"))

    data = load_psi(zp, PSI_TAG)
    if not data:
        raise SystemExit("no routines recovered from the psi source.")
    return render_from_psi(data, zp, allow_vintage_mismatch=a.allow_vintage_mismatch,
                           per_quarter_cbar=(a.per_quarter_cbar == "keep"))


def render_from_psi(data: dict, zp: pathlib.Path, allow_vintage_mismatch: bool = False,
                    per_quarter_cbar: bool = False):
    """The --from-psi pass on psi already loaded ({routine: load_psi entry}); `zp` is where it came
    from, searched for the psi_starts sidecar that records beta/T. Split from main() so the pass
    can be run on psi held in memory."""
    ZERO_PRINTS.clear()
    # Dead firm-quarters: which rows the solve used, and which the tables describe.
    apply_null_fq(data)

    # The ridge table is computed here from the psi archive while every cost figure comes from
    # cost_params. Those must be the same design or the table is a chimera — see
    # check_psi_matches_solve() for the 2026-08-17 case where they were not.
    mismatch = check_psi_matches_solve(data)
    if mismatch:
        msg = ("psi archive does not match the solve that produced cost_params:\n  "
               + "\n  ".join(mismatch)
               + "\n\nThe ridge columns would describe a different psi than the cost columns."
                 "\nFetch the psi that the solve actually consumed, or pass"
                 " --allow-vintage-mismatch to emit anyway (not reportable).")
        if not allow_vintage_mismatch:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg + "\n")

    # A cost_params that predates the rule is restated without the dead firm-quarters (the rows
    # the tables describe), and the record the default pass restates it from is written.
    restate_table_mask(data, zp.name)

    ridge, rows = collect(data)

    for E in sorted(ridge):
        s = ridge[E]
        if s["n_dead_rows"]:
            print(f"  E{E}: {s['n_loaded']:,} psi rows - {s['n_dead_rows']:,} in "
                  f"{s['n_dead_fq']:,} dead firm-quarters ({s['null_mode']}) = {s['n_all']:,}")
        print(f"  E{E}: n={s['n']:,} of {s['n_all']:,} rows (dpsi2==0: {s['n_d2_zero']:,}, "
              f"non-finite: {s['n_nonfinite']:,}; solve n_rows B+D: "
              f"{s['n_ineq'] if s['n_ineq'] is None else format(s['n_ineq'], ',')}) "
              f"({s['n_shards']} shards, tag "
              f"'{data[E].get('psi_tag') or '(none)'}')  corr={s['corr']:.8f}  "
              f"cond={s['cond']:,.0f}  median ratio={s['ratio_median']:.5f} "
              f"(IQR {s['iqr_pct']:.1f}%)  mean={s['ratio_mean']:.5f} (CV {s['cv']:.1f}%, "
              f"top-10 rows {s['cv_top10_ss_pct']:.1f}% of its SS)  "
              f"1-R2={s['one_minus_r2']:.2e}  "
              f"across-D={s['across_delta_pct']:.1f}%  starts={s['n_starts']}")

    # The ridge tables are the only TABLES this path writes (plus, for a cost_params that predates
    # the dead firm-quarter rule, the record restate_table_mask wrote above). They are per routine
    # over the pooled design, which needs the psi itself — cost_params stores its ridge fields per
    # firm type, so the pooled columns (the ratio's median/IQR/mean/CV, the shared #Delta) cannot
    # be recovered from JSON. The two cost tables come from cost_params via the default path and
    # are left untouched here.
    # beta and T of the psi these tables are computed from: the sidecar sits in the same archive
    # (or folder) as the shards, and is matched to the solve by its per-start rbar_f.
    prov = {E: _run_discount(E, d.get("psi_tag") or "", d["cost"], extra=[zp])
            for E, d in sorted(data.items())}
    d8 = _discount_report("tab_bbl_ridge_diagnostic",
                          _routine_entries([E for E in ROUTINE_ORDER if E in ridge], prov))
    # The by-start table exists only for a multi-start psi: with one forward curve there is no
    # start_q column to group on, and a one-row "pooled" table would restate the table above.
    # Computed first because the ridge table carries its lowest within-quarter condition index.
    by_start = {E: ridge_by_start(d["psi_eq"], d["psi_dev"]) for E, d in sorted(data.items())}
    by_start = {E: v for E, v in by_start.items() if v}
    tex = {"tab_bbl_ridge_diagnostic.tex": build_ridge(ridge, disc=d8[0], by_start=by_start)}
    md = {"tab_bbl_ridge_diagnostic.md": md_ridge(ridge, disc=d8[1], by_start=by_start)}
    if by_start:
        cost = {E: d["cost"] for E, d in data.items()}
        for E, blocks in by_start.items():
            qs = [q for q in blocks if q != "pooled"]
            rb = [blocks[q]["ratio_median"] for q in qs]
            n_q = sum(blocks[q]["n"] for q in qs)
            print(f"  E{E}: {len(qs)} launch quarters; median dpsi4/dpsi2 in "
                  f"[{min(rb):.5f}, {max(rb):.5f}] (spread {max(rb)-min(rb):.5f}), "
                  f"pooled {blocks['pooled']['ratio_median']:.5f}; quarters' n sum to "
                  f"{n_q:,} vs pooled {blocks['pooled']['n']:,}")
        d10 = _discount_report("tab_bbl_ridge_by_start",
                               _routine_entries([E for E in ROUTINE_ORDER if E in by_start], prov))
        dead_q = {E: (d.get("null") or {}).get("dead_q") or {} for E, d in data.items()}
        for E, qs in sorted(dead_q.items()):
            for q, ks in sorted(qs.items()):
                print(f"  E{E} {q}: every {'/'.join(ks)} firm-quarter is dead -- Frac. viol. and "
                      f"c-bar dashed in that row")
        tex["tab_bbl_ridge_by_start.tex"] = build_ridge_by_start(by_start, cost, disc=d10[0],
                                                                 dead_q=dead_q,
                                                                 cbar=per_quarter_cbar)
        md["tab_bbl_ridge_by_start.md"] = md_ridge_by_start(by_start, cost, disc=d10[1],
                                                            dead_q=dead_q,
                                                            cbar=per_quarter_cbar)
    else:
        print("  no start_q column in psi — single-start design, "
              "tab_bbl_ridge_by_start not written.")

    # APPENDIX: the violated share by the direction of the deviation (the user adds the \input).
    ud = violated_by_sign(data)
    if ud:
        dud = _discount_report("tab_bbl_violated_by_sign",
                               _routine_entries([E for E in ROUTINE_ORDER if E in ud], prov))
        dead_any = any((d.get("null") or {}).get("dropped") for d in data.values())
        tex["tab_bbl_violated_by_sign.tex"] = build_violated_by_sign(ud, disc=dud[0],
                                                                     dead=dead_any)
        md["tab_bbl_violated_by_sign.md"] = md_violated_by_sign(ud, disc=dud[1], dead=dead_any)

    for name, txt in tex.items():
        _write(name, txt)
    for name, txt in md.items():
        _write(name, txt + "\n")
    # The numbers these tables print, from the records they were rendered from. The Frac. viol.
    # of a launch quarter is _solve_at's, with the arguments build_ridge_by_start passes it.
    tags = {E: _cost_tag(d.get("cost_solve") or d["cost"], d.get("psi_tag"))
            for E, d in data.items()}
    common = dict(theta=_theta_of(rows), provenance=_prov_numbers(prov, tags))
    # Every ridge_stats key stays in the record, the columns the trimmed table no longer prints
    # included; the lowest within-quarter condition index it now prints is added beside them.
    nums = {"tab_bbl_ridge_diagnostic": dict(
        common, routines={f"E{E}": dict(_ridge_numbers(s),
                                        cond_within_min=_within_cond_min(by_start.get(E))[0],
                                        cond_within_min_quarter=_within_cond_min(
                                            by_start.get(E))[1])
                          for E, s in sorted(ridge.items())})}
    if by_start:
        nums["tab_bbl_ridge_by_start"] = dict(common, routines={
            f"E{E}": {q: dict(_ridge_numbers(s),
                              dead_types=list((dead_q.get(E) or {}).get(q, ())),
                              frac_viol=_solve_at(cost.get(E) or {}, None if q == "pooled" else q,
                                                  (dead_q.get(E) or {}).get(q, ()))[0])
                      for q, s in blocks.items()}
            for E, blocks in sorted(by_start.items())})
    if ud:
        nums["tab_bbl_violated_by_sign"] = dict(common, routines={f"E{E}": v
                                                                 for E, v in sorted(ud.items())})
    _write_numbers(_numbers_tag(tags.values()), nums, source=f"psi {zp.name}")
    # Echoed as well as written: on the cluster the SLURM log is the only copy anyone reads.
    for name, txt in md.items():
        print(f"\n----- {name} -----\n{txt}")
    print("\n\\input{tab_bbl_ridge_diagnostic.tex}  (needs booktabs + xltabular)")
    if "tab_bbl_ridge_by_start.tex" in tex:
        print("\\input{tab_bbl_ridge_by_start.tex}")
    if "tab_bbl_violated_by_sign.tex" in tex:
        print("\\input{tab_bbl_violated_by_sign.tex}  % appendix")
    _report_zero_prints("ridge tables (T8, T10) and the appendix up/down table")


if __name__ == "__main__":
    main()
