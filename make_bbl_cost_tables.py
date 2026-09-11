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
  2. tab_bbl_cbar.tex              -- c_bar and the criterion-health row. The estimand when the
                                      split is NOT identified; a derived summary when it is.
  3. tab_bbl_ridge_diagnostic.tex  -- the collinearity evidence: corr, cond, ratio dispersion,
                                      1-R^2, and the across-Delta spread that rules out a
                                      subsample fix.  (--from-psi)
  4. tab_bbl_ridge_by_start.tex    -- the same design measured LAUNCH QUARTER BY LAUNCH QUARTER,
                                      plus a pooled row: this is where a broken ridge shows up as
                                      rbar_f moving across quarters.  (--from-psi, multi-start
                                      psi only -- it needs the start_q column)
All are \input-ready in the V_Main house style (spacing + xltabular at \textwidth + booktabs,
caption/label inside the table, notes in \endlastfoot) -- see _wrap() and polfunc_k4.tex.
Also writes markdown twins next to identification_notes.md for the notes document:
  tab_bbl_cost_identified.md, tab_bbl_cbar.md, tab_bbl_ridge_diagnostic.md,
  tab_bbl_ridge_by_start.md

WHICH PASS PRODUCES THE REPORTED TABLES. Both, and both on the cluster: bbl_job.sh's `tables`
step runs the default pass and then the --from-psi pass against the shards in ${CF_COST_FWD},
and the .tex/.md they write into the BBL step folder are what cluster_archive.sh --set bbl
brings home. The two passes write DISJOINT files, so neither can overwrite the other's.

--from-psi is not a local convenience that could be dropped in favour of reading cost_params:
the ridge tables are per ROUTINE over the pooled design, while cost_params stores its ridge
fields per FIRM TYPE, so the pooled columns (Ratio, CV%, the shared #Delta) have no JSON
source -- see the comment at the head of that branch. Run it locally only to inspect a
downloaded psi_cost.zip; the numbers in the paper come from the cluster pass.

Usage:
  python make_bbl_cost_tables.py                       # cost_params only (vintage-safe)
  python make_bbl_cost_tables.py --from-psi            # + the ridge tables, from psi_cost.zip
  python make_bbl_cost_tables.py --from-psi --psi-dir "$CF_COST_FWD"   # on the cluster
  python make_bbl_cost_tables.py --psi-tag _ms8        # pin one multi-start vintage
"""

from __future__ import annotations

import argparse
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

# A quarterly rate -> pp/yr uses the code's own x400 convention (bbl_fwd_sim.jl:473).
PP_YR = 400.0

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


def ridge_stats(eq: pd.DataFrame, dev: pd.DataFrame, is_B: bool | None = None,
                start: str | None = None) -> dict:
    """Collinearity of the two columns that carry omega and zeta in the DIFFERENCED design.

    is_B selects a single firm type (True=B, False=D); None pools both, which is what the
    reported ridge table uses. The per-type call exists for check_psi_matches_solve(), because
    cost_params stores its ridge fields per type.

    start selects one launch quarter of a multi-start psi; None pools every quarter, which is
    what the pooled row of the by-start table and the whole of the per-routine ridge table use.
    """
    m = dev.merge(eq, on=_merge_keys(eq, dev), suffixes=("_d", "_e"))
    if is_B is not None:
        m = m[m["is_B"].to_numpy().astype(bool) == is_B]
    if start is not None and "start_q" in m.columns:
        m = m[m["start_q"].astype(str) == str(start)]
    d2 = (m["psi2_omega_e"] - m["psi2_omega_d"]).to_numpy(float)
    d4 = (m["psi4_zeta_e"] - m["psi4_zeta_d"]).to_numpy(float)
    ok = np.isfinite(d2) & np.isfinite(d4) & (d2 != 0)
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

    return dict(
        n=int(len(d2)), n_firms=int(m["firm"].nunique()),
        n_shock=int(m.loc[ok, "shock"].nunique()) if "shock" in m.columns else -1,
        corr=float(np.corrcoef(d2, d4)[0, 1]),
        cond=cond,
        cond_raw=float(np.linalg.cond(X)),
        rbar=float(np.median(ratio)), ratio_mean=float(ratio.mean()),
        cv=float(100 * ratio.std() / abs(ratio.mean())),
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
    forward curve every quarter shares one rbar_f and the pooled ratio is a constant, so the
    corr/cond columns repeat down the page. Quarter-by-quarter movement in rbar_f is the
    identifying variation, and it is visible here before it is visible in any estimate.
    """
    qs = start_quarters(eq)
    if not qs:
        return {}
    out = {}
    for q in qs:
        try:
            out[q] = ridge_stats(eq, dev, start=q)
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            # One empty or degenerate quarter must not cost the other 35 their row.
            print(f"  [skip] start {q}: {type(exc).__name__}: {exc}")
    out["pooled"] = ridge_stats(eq, dev)
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
    be written per routine or per firm-type block. All reported blocks must agree before the
    identified presentation fires: a mixed state is reported and takes the conservative branch,
    which is the one whose notes are still true when a block is ridge-degenerate.
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
        print("  [note] identified_split disagrees across blocks "
              f"({sum(verdicts)}/{len(verdicts)} true) — reporting the ridge-degenerate "
              "presentation, whose notes hold in both regimes.")
        return False
    return all(verdicts)


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
    """
    problems, checked = [], 0
    for E, d in sorted(data.items()):
        for key, _lbl in BLOCKS:
            blk = (d["cost"] or {}).get(key)
            if not blk:
                continue
            st = ridge_stats(d["psi_eq"], d["psi_dev"], is_B=(key == "B"))
            for field, mine in (("rbar_f", st["rbar"]),
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
        ridge[E] = st
        for key, lbl in BLOCKS:
            blk = d["cost"].get(key)
            if not blk:
                continue
            w, z = float(blk["omega"]), float(blk["zeta"])
            # Prefer the solve's own r_bar/c_bar (computed on the same design it fitted); fall
            # back to the psi-derived value for cost_params written before those fields existed.
            rbar = float(blk.get("rbar_f", st["rbar"]))
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
            ))
    return ridge, rows


# ── latex helpers (V_Main house style) ──────────────────────────────────────
def _f(x, nd=4, dash="---"):
    return dash if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"${x:.{nd}f}$"


def _sci(x, nd=2):
    r"""1.23e-05 -> $1.23\times10^{-5}$. Python's own exponent form typesets as `1.23e - 05`
    inside math mode: the `e` italicises as a variable and the exponent's minus picks up binary-
    operator spacing."""
    m, e = f"{x:.{nd}e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def _num(x, nd=0):
    r"""Thousands separator that survives math mode: `$5,122$` typesets as `5, 122` because a
    comma is punctuation in math. Kept as text, so the digits stay upright like the other
    counts in these tables."""
    return f"{x:,.{nd}f}"


def _wrap(body, col_fmt, caption, label, header, footnote, ncols):
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
    """
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
        rf"{{\scriptsize {footnote}}} \\",
        r"\endlastfoot",
        *body,
        r"\end{xltabular}",
        r"\end{spacing}",
        "",
    ])


def build_ridge(ridge):
    # Nine columns at \footnotesize in a portrait text block: the headers are kept to one short
    # word each and every symbol is defined in the note, because a spelled-out header like
    # corr$(\Delta\psi_2,\Delta\psi_4)$ wraps to three lines and collides with its neighbour.
    # The condition number is "Cond." and NOT $\kappa$ — the paper already spends $\kappa$ on the
    # firm type (B/D), and this table is read alongside cost estimates indexed by it.
    col_fmt = (r">{\raggedright\arraybackslash}p{1.7cm} "
               r"*{8}{>{\centering\arraybackslash}X}")
    header = (r"Routine & $n$ & Firms & $\#\Delta$ & Corr & Cond. & Ratio & CV\% & $1-R^2$")
    body = []
    for E in sorted(ridge):
        s = ridge[E]
        body.append(" & ".join([
            rc.est_ref(E), _num(s["n"]), str(s["n_firms"]), str(s["n_shock"]),
            f"${s['corr']:.8f}$", _num(s["cond"]),
            _f(s["ratio_mean"], 5), f"${s['cv']:.2f}$", _sci(s["one_minus_r2"]),
        ]) + r" \\")
    # Interpretation lives in the surrounding prose, not here: the note states only what the
    # table is. The across-Delta spread that used to close it is still measured in ridge_stats
    # and echoed to the console, so the figures quoted in the text stay checkable.
    foot = (
        r"\textit{Notes:} Collinearity of the two $\psi$ columns that carry $\omega$ and $\zeta$ "
        r"in the \emph{differenced} design $g=\Delta\psi_1-\omega\Delta\psi_2-\gamma'\Delta\psi_3"
        r"-(1+\zeta)\Delta\psi_4$ of \eqref{eq:16}."
    )
    return _wrap(body, col_fmt, r"$\psi_2/\psi_4$ Ridge",
                 "tab:bbl_ridge_diagnostic", header, foot, 9)


def _cond_cell(x, sci_above=1e6):
    r"""A condition index as a table cell. Above ~1e6 it switches to $a\times10^{b}$: within a
    single launch quarter the two psi columns are collinear to machine precision (psi4 is
    rbar_f*psi2 there by construction), so the index runs to 1e15 and a thousands-separated
    integer is 20 characters of noise in an X column sized for four."""
    if x is None or not np.isfinite(x):
        return "---"
    return _sci(float(x)) if abs(float(x)) >= sci_above else _num(float(x))


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


def _solve_at(rec: dict, q: str | None) -> tuple:
    r"""-> (frac_bind, c_bar) for one routine at one launch quarter, pooled across firm types.

    q=None takes the block's own pooled figures. A per-quarter figure exists only when the solve
    refitted inside the quarter (ridge_by_start's within-start refit); without that record the
    two columns are dashed rather than repeating the pooled number down 36 rows, which would read
    as 36 measurements of something that was measured once.

    Pooled across B and D by the inequality count, the same weight the criterion itself gives
    each block, so the pooled row equals what a single-block solve of the union would report if
    the two types shared a theta.
    """
    fb_num = cb_num = w_tot = 0.0
    seen = False
    for key, _lbl in BLOCKS:
        blk = (rec or {}).get(key)
        if not blk:
            continue
        src = blk if q is None else _per_start(blk).get(q)
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


def build_ridge_by_start(by_start: dict, cost: dict):
    r"""The ridge measured one LAUNCH QUARTER at a time, one panel per routine.

    Separate from tab:bbl_ridge_diagnostic rather than added to it as extra rows: that table is
    the pooled statement (one row per routine) the text cites, and a reader checking whether the
    multi-start design broke the ridge is asking a different question -- does rbar_f MOVE across
    quarters -- which needs the quarters down the rows and the routines split into panels.
    """
    col_fmt = (r">{\raggedright\arraybackslash}p{1.7cm} "
               r"*{6}{>{\centering\arraybackslash}X}")
    header = (r"Start & $n$ & $\bar r^f$ & Corr & Cond. & Frac.\ viol. & $\bar c$")
    ncol = 7
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
            fb, cb = _solve_at(cost.get(E) or {}, None if q == "pooled" else q)
            lbl = r"\textit{Pooled}" if q == "pooled" else q
            if q == "pooled":
                body.append(r"\addlinespace[0.3ex]")
            body.append(" & ".join([
                lbl, _num(s["n"]), _f(s["rbar"], 5),
                f"${s['corr']:.6f}$", _cond_cell(s["cond"]),
                _f(fb, 3), _f(cb, 4),
            ]) + r" \\")
    foot = (
        r"\textit{Notes:} The differenced design of \eqref{eq:16} measured separately at each "
        r"forward-curve launch quarter, pooling firm types. $\bar r^f$ is the median "
        r"$\Delta\psi_4/\Delta\psi_2$ within the quarter, i.e. the risk-free level at which that "
        r"quarter's deviations are priced; \emph{Corr} and \emph{Cond.} are the correlation and "
        r"the Belsley-Kuh-Welsch condition index of $[\Delta\psi_2\ \Delta\psi_4]$ with the "
        r"columns scaled to unit length. Movement in $\bar r^f$ down a panel is the variation "
        r"that separates $\omega$ from $\zeta$: a design with one forward curve holds it fixed "
        r"and leaves only the combination $\bar c=\omega+\bar r^f\zeta$ identified. "
        r"\emph{Frac.\ viol.} and $\bar c$ come from the solve; they are dashed where it did not "
        r"fit that quarter separately, and pooled across firm types by the inequality count."
    )
    return _wrap(body, col_fmt, r"$\psi_2/\psi_4$ Ridge by Forward-Curve Launch Quarter",
                 "tab:bbl_ridge_by_start", header, foot, ncol)


def md_ridge_by_start(by_start: dict, cost: dict):
    L = []
    for E in [E for E in ROUTINE_ORDER if E in by_start and by_start[E]]:
        blocks = by_start[E]
        L += [f"**E{E}**", "",
              "| Start | n | rbar_f | Corr | Cond. | Frac. viol. | c-bar |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for q in [k for k in blocks if k != "pooled"] + ["pooled"]:
            s = blocks[q]
            fb, cb = _solve_at(cost.get(E) or {}, None if q == "pooled" else q)
            L.append(f"| {'*pooled*' if q == 'pooled' else q} | {s['n']:,} | {s['rbar']:.5f} | "
                     f"{s['corr']:.6f} | {s['cond']:,.3g} | "
                     f"{'--' if fb is None else f'{fb:.3f}'} | "
                     f"{'--' if cb is None else f'{cb:.4f}'} |")
        L.append("")
    L += ["The differenced design measured per forward-curve launch quarter, firm types pooled. "
          "*rbar_f* = median dpsi_4/dpsi_2 within the quarter; movement in it down a panel is "
          "the variation that separates omega from zeta. *Frac. viol.* and *c-bar* come from "
          "the solve and are dashed where it did not fit that quarter separately."]
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
    ("psi3_gamma_indice_basileia_lag",      r"Basel Index ($t-1$)"),
]


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


def build_cbar_panels(rows, identified=False):
    r"""\bar c^\kappa and the health metric that governs whether it may be read.

    Split out of the parameter table deliberately: frac_bind is the gate on quoting any cost
    figure at all, so it belongs on one page with the cost figure rather than buried under eight
    rows of coefficients.

    `identified` says whether the solve separated \omega from \zeta. It changes what \bar c IS,
    not how it is computed: with a single forward curve it is the one direction the design pins
    down and therefore the estimand; with launch-quarter variation \omega and \zeta are each
    identified and \bar c becomes a derived summary of them, evaluated at the mean forward rate.
    The notes say which, because the unqualified claim is false in the other regime.
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

        body.append(r"$\bar r^f$ & " + cells(lambda r: f"${r['rbar']:.5f}$") + r" \\")
        body.append(r"\addlinespace[0.3ex]")
        body.append(rf"$\hat{{\bar c}}^{{\mathrm{{{kappa}}}}}$ & "
                    + cells(lambda r: f"${r['cbar']:.4f}$") + r" \\")
        body.append(" & " + cells(lambda r: f"$({r['cbar_se']:.4f})$"
                                  if r.get("cbar_se") is not None else "") + r" \\")
        body.append(r"\quad 95\% CI & "
                    + cells(lambda r: rf"{{\scriptsize $[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]$}}"
                            if r.get("ci_lo") is not None else "---") + r" \\")
        body.append(r"\addlinespace[0.4ex]")
        body.append(FRAC_BIND_LABEL + " & "
                    + cells(lambda r: f"${r['frac_bind']:.3f}$"
                            if r.get("frac_bind") is not None else "---") + r" \\")
    common = (
        r"Quarterly units, and excluding the $\boldsymbol{\gamma}'\boldsymbol{Z}$ shifters. "
        r"The standard error is a firm-block bootstrap standard deviation and the interval is "
        r"the subsampling one from Section~\ref{sec:empirical:cost}, which is the appropriate "
        r"route for a criterion that is kinked and potentially set-identified. "
        r"The last row is the share of the firm~$\times$~deviation revealed-preference "
        r"inequalities $g=V(\hat\sigma)-V(\tilde\sigma)\ge 0$ that \emph{fail} at "
        r"$\hat\theta$, see Section~\ref{sec:empirical:cost}."
    )
    if identified:
        foot = (
            r"\textit{Notes:} $\bar c^\kappa = \omega^\kappa + \bar r^f\zeta^\kappa$ is the "
            r"marginal cost of deposits evaluated at the mean forward risk-free rate. Here it is "
            r"a \emph{derived} summary, not the estimand: simulating each market from its own "
            r"launch quarter's forward curve makes $\bar r^f$ vary across quarters "
            r"(Table~\ref{tab:bbl_ridge_by_start}), which separates $\Delta\psi_2$ from "
            r"$\Delta\psi_4$, so $\omega^\kappa$ and $\zeta^\kappa$ are each identified and are "
            r"reported as the estimates in Table~\ref{tab:bbl_cost_identified}. This table "
            r"restates them at one rate for comparability with the single-curve design and with "
            r"the marginal costs elsewhere in the paper. " + common
        )
    else:
        foot = (
            r"\textit{Notes:} $\bar c^\kappa = \omega^\kappa + \bar r^f\zeta^\kappa$ is the "
            r"marginal cost of deposits at the mean forward risk-free rate, and the only cost "
            r"object this design identifies: $\Delta\psi_2$ and $\Delta\psi_4$ are collinear at "
            r"$\mathrm{corr}>0.9999$ (Table~\ref{tab:bbl_ridge_diagnostic}), so $\omega^\kappa$ "
            r"and $\zeta^\kappa$ are separately unidentified and are reported in "
            r"Table~\ref{tab:bbl_cost_identified} for completeness only. " + common
        )
    caption = (r"BBL Marginal Cost $\bar c^\kappa$ at the Mean Forward Rate" if identified
               else r"BBL Identified Marginal Cost $\bar c^\kappa$ and Criterion Health")
    return _wrap(body, col_fmt, caption, "tab:bbl_cbar", header, foot, ncol)


def md_cbar_panels(rows, identified=False):
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
        row("rbar_f", lambda r: f"{r['rbar']:.5f}")
        row(f"c-bar^{kappa}", lambda r: f"{r['cbar']:.4f} ({r['cbar_se']:.4f})")
        row("95% CI", lambda r: f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            if r.get("ci_lo") is not None else "--")
        row(FRAC_BIND_LABEL_MD, lambda r: f"{r['frac_bind']:.3f}")
    fbs = [r["frac_bind"] for r in rows if r.get("frac_bind") is not None]
    lead = ("c-bar^kappa = omega^kappa + rbar_f * zeta^kappa, a DERIVED summary of the two "
            "identified parameters at the mean forward rate: launch-quarter variation in rbar_f "
            "separates them, so the estimates are omega and zeta themselves (see the by-start "
            "ridge table). "
            if identified else
            "c-bar^kappa = omega^kappa + rbar_f * zeta^kappa, the only cost object this design "
            "identifies; omega and zeta separately are not (see the ridge table). ")
    L += ["",
          lead +
          "SE is a "
          "firm-block bootstrap SD (200 reps); the CI is the subsampling sqrt(n)-rate quantile "
          "interval (b = n^(2/3) firms, 200 reps). The last row is the share of the 50 "
          "firm × deviation revealed-preference inequalities per firm that FAIL at theta-hat — "
          "deviations the model says would have raised the firm's value, i.e. price moves the "
          "bank should have made and did not. A genuine best response implies a share near 0; "
          f"here it is [{min(fbs):.3f}, {max(fbs):.3f}], and 1/2 is the mechanical value of a "
          "symmetric ± grid around a policy with no interior turning point (exactly one of each "
          "± pair fails for ANY theta). It carries no information about fit, so these magnitudes "
          "are diagnostics rather than estimates."]
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
    return rf"{{\scriptsize $[{lo:.3f},{hi:.3f}]${mark}}}"


def build_identified_panels(rows, identified=False):
    r"""Estimates table in the paper's own layout: parameters down the rows, estimation routines
    across the columns as \ref{estimation:*} (rendered (III)-(VI)), and one panel per firm type.

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

    def line(label, get, fmt="{:.3f}", bold=False, se_get=None, se_fmt="{:.3f}", stars=False):
        """One estimate row, plus a parenthesised SE row underneath when se_get is given."""
        cells = []
        for E in Es:
            r = by.get((E, kappa))
            v = get(r) if r else None
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("---")
            else:
                s = fmt.format(v)
                if bold:
                    cells.append(r"\textbf{" + s + "}")
                elif stars:
                    cells.append(f"${s}^{{{_stars(v, se_get(r) if se_get else None)}}}$")
                else:
                    cells.append(f"${s}$")
        out = [label + " & " + " & ".join(cells) + r" \\"]
        if se_get is not None:
            ses = []
            for E in Es:
                r = by.get((E, kappa))
                sv = se_get(r) if r else None
                ses.append(f"$({se_fmt.format(sv)})$" if sv is not None
                           and np.isfinite(sv) else "")
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
    if identified:
        # The interval, not the star, is the claim in this branch — said once, at the top, so a
        # reader who scans only the numbers still meets it. The full statement stays in the Notes.
        body.append(
            rf"\multicolumn{{{ncol}}}{{l}}{{\scriptsize\itshape Intervals invert the criterion "
            rf"at a subsampled critical value; $\dagger$ marks a profile-window endpoint "
            rf"(see Notes).}} \\")
    else:
        # The kinked-criterion caveat, in the table itself and not only mid-footnote: the reader
        # meets it before scanning any stars. The full statement stays in the Notes.
        body.append(
            rf"\multicolumn{{{ncol}}}{{l}}{{\scriptsize\itshape Significance stars are shown by "
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
                     stars=not identified)
        if identified:
            body += ci_line(r"\quad 95\% CI", "ci_omega")
        body += line(rf"$\hat\zeta^{{\mathrm{{{kappa}}}}}$",
                     lambda r: r["zeta"], fmt="{:.2f}",
                     se_get=lambda r: r.get("zeta_se"), se_fmt="{:.1f}",
                     stars=not identified)
        if identified:
            body += ci_line(r"\quad 95\% CI", "ci_zeta")
        body.append(r"\addlinespace[0.3ex]")
        # The gamma block is headed by the bare symbol, flush left like the other parameters;
        # the shifters it loads on are indented beneath it.
        body.append(rf"$\hat{{\boldsymbol{{\gamma}}}}^{{\mathrm{{{kappa}}}}}$"
                    + " & " * len(Es) + r" \\")
        for key, lab in Z_LABELS:
            body += line(r"\hspace{1em}" + lab,
                         (lambda k: lambda r: (r["gamma"] or {}).get(k))(key),
                         se_get=(lambda k: lambda r: (r["gamma_se"] or {}).get(k))(key),
                         stars=not identified)
        # Regression information, separated from the parameters as in the sleepiness tables.
        # (rbar^f is not a parameter of eq:8 — it lives with c-bar, which it defines.)
        body.append(r"\midrule")
        body += line(r"Firms", lambda r: r["n_firms"], fmt="{:.0f}")
        body += line(r"Inequalities $n$", lambda r: r["n"], fmt="{:,.0f}")

    common = (
        r"\textit{Notes:} Deposit-servicing marginal cost parameters of \eqref{eq:8} from the "
        r"BBL moment-inequality problem \eqref{eq:16}, "
        r"estimated separately by firm type; columns are the estimation routines "
        r"of Section~\ref{sec:empirical:sleep}. Quarterly units; the cost-shifter ratios are in "
        r"percentage points and the Basel index a fraction, both lagged one quarter. "
    )
    if identified:
        foot = (
            common +
            r"$\omega^\kappa$ and $\zeta^\kappa$ are separately identified here: each market is "
            r"simulated from the forward curve of its own launch quarter, so $\bar r^f$ varies "
            r"across quarters and $\Delta\psi_2$ and $\Delta\psi_4$ are no longer collinear "
            r"(Table~\ref{tab:bbl_ridge_by_start}). The bracketed interval is the "
            r"\eqref{eq:16} criterion inverted at a subsampled critical value, which is the "
            r"appropriate route for a kinked and potentially set-identified criterion; "
            r"$\dagger$ marks an endpoint at the edge of the profile window, i.e. a limit of the "
            r"search rather than a boundary of the identified set, and \textit{empty} means the "
            r"criterion was never within the critical value at the reported level. Parenthesised "
            r"figures are firm-block bootstrap SDs, descriptive only. "
            r"$\bar c^\kappa=\omega^\kappa+\bar r^f\zeta^\kappa$ in "
            r"Table~\ref{tab:bbl_cbar} restates the pair at the mean forward rate."
        )
    else:
        foot = (
            common +
            r"Standard errors: firm-block bootstrap SDs, descriptive only, see "
            r"Section~\ref{sec:empirical:cost} and diagnostics in "
            r"Table~\ref{tab:bbl_ridge_diagnostic}. "
            r"None of these parameters is separately identified. The "
            r"table is reported for completeness; the estimand is $\bar c^\kappa$ in "
            r"Table~\ref{tab:bbl_cbar}. Significance markers use the normal approximation to the "
            r"bootstrap SD and are shown by convention only --- the criterion in \eqref{eq:16} is "
            r"kinked and potentially set-identified, so that reference distribution does not apply "
            r"and they should not be read as tests. *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
        )
    return _wrap(body, col_fmt,
                 r"BBL Deposit-Servicing Cost Parameters, by Firm Type",
                 "tab:bbl_cost_identified", header, foot, ncol)


def _md_ci(ci):
    if not ci:
        return "--"
    if ci.get("empty"):
        return "*empty*"
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if lo is None or hi is None:
        return "--"
    return (f"[{lo:.3f}, {hi:.3f}]"
            + ("+" if (ci.get("truncated_lo") or ci.get("truncated_hi")) else ""))


def md_identified_panels(rows, identified=False):
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
        row(f"omega^{kappa}", lambda r: f"{r['omega']:.3f}"
                                        f"{'' if identified else _stars(r['omega'], r['omega_se'])} "
                                        f"({r['omega_se']:.3f})")
        if identified:
            row("  95% CI", lambda r: _md_ci(r.get("ci_omega")))
        row(f"zeta^{kappa}", lambda r: f"{r['zeta']:.2f}"
                                       f"{'' if identified else _stars(r['zeta'], r['zeta_se'])} "
                                       f"({r['zeta_se']:.1f})")
        if identified:
            row("  95% CI", lambda r: _md_ci(r.get("ci_zeta")))
        row(f"**gamma^{kappa}**", lambda r: "")
        for key, lab in Z_LABELS:
            row("  " + lab.replace("($t-1$)", "(t-1)"),
                (lambda k: lambda r: (
                    f"{(r['gamma'] or {}).get(k, float('nan')):.3f}"
                    f"{'' if identified else _stars((r['gamma'] or {}).get(k), (r['gamma_se'] or {}).get(k))} "
                    f"({(r['gamma_se'] or {}).get(k, float('nan')):.3f})"))(key))
        row("*Firms*", lambda r: f"{r['n_firms']}")
        row("*Inequalities n*", lambda r: f"{r['n']:,}")
    if identified:
        L += ["",
              "Columns are estimation routines. **omega and zeta are separately identified "
              "here**: each market is simulated from its own launch quarter's forward curve, so "
              "rbar_f varies across quarters and dpsi_2/dpsi_4 are no longer collinear (see the "
              "by-start ridge table). The bracketed 95% CI inverts the eq:16 criterion at a "
              "subsampled critical value — the appropriate route for a kinked, potentially "
              "set-identified criterion; a trailing + marks a profile-window endpoint (a limit "
              "of the search, not a set boundary) and *empty* means the criterion was never "
              "within the critical value. Parenthesised figures are firm-block bootstrap SDs, "
              "descriptive only. c-bar restates the pair at the mean forward rate."]
    else:
        L += ["",
              "Columns are estimation routines. SEs in parentheses are firm-block bootstrap SDs "
              "(200 reps). **None of these parameters is separately identified** — omega, zeta and "
              "gamma slide freely along the ridge (one block returns omega>0 with zeta<0 while c-bar "
              "barely moves); the estimand is c-bar, reported separately. Stars use the normal "
              "approximation to the bootstrap SD and are shown by convention only: the criterion is "
              "kinked and potentially set-identified, so they should not be read as tests. "
              "*** p<0.01, ** p<0.05, * p<0.1."]
    return "\n".join(L)


def md_ridge(ridge):
    # Short headers on purpose: pandoc/xelatex renders this at 11pt in a 1in-margin portrait
    # page, and a header like "corr(dpsi2,dpsi4)" collides with its neighbour. Symbols are
    # defined in the line below the table instead.
    L = ["| Routine | n | Firms | Delta | Corr | Cond. | Ratio | CV% | 1-R2 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for E in sorted(ridge):
        s = ridge[E]
        L.append(f"| E{E} | {s['n']:,} | {s['n_firms']} | {s['n_shock']} | {s['corr']:.8f} | "
                 f"{s['cond']:,.0f} | {s['ratio_mean']:.5f} | {s['cv']:.2f} | "
                 f"{s['one_minus_r2']:.2e} |")
    L += ["",
          "*Corr* = corr(Δpsi_2, Δpsi_4); *Cond.* = Belsley-Kuh-Welsch (1980) condition index of "
          "[Δpsi_2 Δpsi_4], i.e. with the columns scaled to unit length, so it measures "
          "collinearity alone and not the columns' units; >30 is the usual threshold. "
          "*Ratio* = mean Δpsi_4/Δpsi_2; *1-R2* = share of Δpsi_4 **not** explained by Δpsi_2 "
          "alone; *n* = firm × deviation pairs; *Delta* = number of distinct perturbations."]
    return "\n".join(L)


def main_from_json():
    """Build both tables from cost_params alone (default). Vintage-safe by construction: every
    column, ridge diagnostics included, comes from the solve's own record of the psi it used."""
    cost = load_cost_only()
    if not cost:
        raise SystemExit(f"no cost_params found in {COST_DIR}")
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
    print(f"  identified_split = {ident}  -> "
          + ("omega/zeta are the estimand; c_bar is derived." if ident
             else "c_bar is the estimand; the omega/zeta split is reported for completeness."))
    # Only the two cost tables come from cost_params. The ridge tables are per-routine over the
    # pooled design and are built from the psi by --from-psi; they are deliberately NOT
    # written here, so a cost-side rebuild cannot restyle or overwrite them.
    tex = {"tab_bbl_cbar.tex": build_cbar_panels(rows, ident),
           "tab_bbl_cost_identified.tex": build_identified_panels(rows, ident)}
    md = {"tab_bbl_cbar.md": md_cbar_panels(rows, ident),
          "tab_bbl_cost_identified.md": md_identified_panels(rows, ident)}
    for name, txt in tex.items():
        _write(name, txt)
    for name, txt in md.items():
        _write(name, txt + "\n")
    # The markdown twins go to stdout as well. Nothing is downloaded from the cluster, so the
    # SLURM log has to BE the deliverable: a table that only exists as a file in the step folder
    # is a table nobody can read without another transfer.
    for name, txt in md.items():
        print(f"\n----- {name} -----\n{txt}")
    print("\n\\input{tab_bbl_cbar.tex}            % tab:bbl_cbar")
    print("\\input{tab_bbl_cost_identified.tex}  % tab:bbl_cost_identified")
    print("\\input{tab_bbl_ridge_diagnostic.tex} % tab:bbl_ridge_diagnostic (--from-psi)")
    print("\\input{tab_bbl_ridge_by_start.tex}   % tab:bbl_ridge_by_start   (--from-psi, _ms)")
    print("(needs booktabs + xltabular, both already in V_Main)")


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
    ap.add_argument("--allow-vintage-mismatch", action="store_true",
                    help="with --from-psi: emit tables even if the psi archive is not the psi "
                         "the solve used. Only for inspecting a known-mixed pair; not reportable.")
    a = ap.parse_args()

    PSI_TAG = a.psi_tag
    COST_DIR = pathlib.Path(a.cost_dir) if a.cost_dir else _resolve_cost_dir(PSI_TAG)
    print(f"cost_params dir: {COST_DIR}")
    print(f"table dests:     {', '.join(str(d) for d in _dests())}")

    if not a.from_psi:
        return main_from_json()

    zp = pathlib.Path(a.psi_dir) if a.psi_dir else pathlib.Path(a.psi_zip)
    if not zp.exists():
        raise SystemExit(f"psi source not found: {zp}\nPass --psi-zip or --psi-dir.")
    print(f"reading {zp.name}"
          + ("/ (loose shards)" if zp.is_dir() else f" ({zp.stat().st_size/1e6:.1f} MB)"))

    data = load_psi(zp, PSI_TAG)
    if not data:
        raise SystemExit("no routines recovered from the psi source.")

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
        if not a.allow_vintage_mismatch:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg + "\n")

    ridge, rows = collect(data)

    for E in sorted(ridge):
        s = ridge[E]
        print(f"  E{E}: n={s['n']:,} ({s['n_shards']} shards, tag "
              f"'{data[E].get('psi_tag') or '(none)'}')  corr={s['corr']:.8f}  "
              f"cond={s['cond']:,.0f}  rbar={s['rbar']:.5f}  1-R2={s['one_minus_r2']:.2e}  "
              f"across-D={s['across_delta_pct']:.1f}%  starts={s['n_starts']}")

    # The ridge tables are the ONLY thing this path writes. They are per routine over the pooled
    # design, which needs the psi itself — cost_params stores its ridge fields per firm type, so
    # the pooled columns (Ratio, CV%, the shared #Delta) cannot be recovered from JSON. The two
    # cost tables come from cost_params via the default path and are left untouched here.
    tex = {"tab_bbl_ridge_diagnostic.tex": build_ridge(ridge)}
    md = {"tab_bbl_ridge_diagnostic.md": md_ridge(ridge)}

    # The by-start table exists only for a multi-start psi: with one forward curve there is no
    # start_q column to group on, and a one-row "pooled" table would restate the table above.
    by_start = {E: ridge_by_start(d["psi_eq"], d["psi_dev"]) for E, d in sorted(data.items())}
    by_start = {E: v for E, v in by_start.items() if v}
    if by_start:
        cost = {E: d["cost"] for E, d in data.items()}
        for E, blocks in by_start.items():
            qs = [q for q in blocks if q != "pooled"]
            rb = [blocks[q]["rbar"] for q in qs]
            print(f"  E{E}: {len(qs)} launch quarters; rbar_f in "
                  f"[{min(rb):.5f}, {max(rb):.5f}] (spread {max(rb)-min(rb):.5f}), "
                  f"pooled {blocks['pooled']['rbar']:.5f}")
        tex["tab_bbl_ridge_by_start.tex"] = build_ridge_by_start(by_start, cost)
        md["tab_bbl_ridge_by_start.md"] = md_ridge_by_start(by_start, cost)
    else:
        print("  no start_q column in psi — single-start design, "
              "tab_bbl_ridge_by_start not written.")

    for name, txt in tex.items():
        _write(name, txt)
    for name, txt in md.items():
        _write(name, txt + "\n")
    # Echoed as well as written: on the cluster the SLURM log is the only copy anyone reads.
    for name, txt in md.items():
        print(f"\n----- {name} -----\n{txt}")
    print("\n\\input{tab_bbl_ridge_diagnostic.tex}  (needs booktabs + xltabular)")
    if "tab_bbl_ridge_by_start.tex" in tex:
        print("\\input{tab_bbl_ridge_by_start.tex}")


if __name__ == "__main__":
    main()
