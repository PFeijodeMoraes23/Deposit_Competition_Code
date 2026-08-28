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

Writes (to ESTIMATION_OUTPUT/Rout/ AND Drafts/Deposit Competition/):
  1. tab_bbl_cost_identified.tex   -- c_bar per routine x firm type, with the raw split shown
                                      struck through as diagnostics only.
  2. tab_bbl_ridge_diagnostic.tex  -- the collinearity evidence: corr, cond, ratio dispersion,
                                      1-R^2, and the across-Delta spread that rules out a
                                      subsample fix.
Both are \input-ready in the V_Main house style (spacing + xltabular at \textwidth + booktabs,
caption/label inside the table, notes in \endlastfoot) -- see _wrap() and polfunc_k4.tex.
Also writes markdown twins next to identification_notes.md for the notes document:
  tab_bbl_cost_identified.md, tab_bbl_ridge_diagnostic.md

Usage:
  python make_bbl_cost_tables.py            # reads psi_cost.zip from BBL_OUTPUT/cluster_raw
  python make_bbl_cost_tables.py --psi-zip PATH
"""

from __future__ import annotations

import argparse
import io
import json
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
COST_DIR = _paths.bbl_output_dir() / "cluster_processed"

ROUTINES = tuple(_routines.LINK_ESTS)
BLOCKS = (("B", r"Brick \& mortar"), ("D", "Digital"))
SPEC, STAGE = _routines.SPEC12_ID, "extended"

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


# ── data ─────────────────────────────────────────────────────────────────────
def load_from_zip(zpath: pathlib.Path):
    """-> {routine: {'psi_eq': df, 'psi_dev': df, 'cost': dict}}. Reads the archive in place so
    400 shard parquets are not duplicated onto disk."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()

        def _read_parquet(n):
            return pd.read_parquet(io.BytesIO(z.read(n)))

        for E in ROUTINES:
            eq_n = [n for n in names if n.endswith(f"psi_eq_E{E}_spec_{SPEC}_{STAGE}.parquet")]
            dv_n = sorted(n for n in names
                          if re.search(rf"psi_dev_E{E}_spec_{SPEC}_{STAGE}_shard\d+of\d+\.parquet$", n))
            # cost_params come from cluster_processed, NOT the archive: the archived copies
            # predate the c_bar/ridge fields that bbl_solve.py now persists.
            # Fall back to the archived copy so the script still runs on an untouched tree.
            cp_live = COST_DIR / f"cost_params_E{E}_spec_{SPEC}_{STAGE}.json"
            cp_n = [n for n in names if n.endswith(f"cost_params_E{E}_spec_{SPEC}_{STAGE}.json")]
            if not (eq_n and dv_n and (cp_live.exists() or cp_n)):
                print(f"  [skip] E{E}: eq={len(eq_n)} dev={len(dv_n)} cost={cp_live.exists() or len(cp_n)}")
                continue
            cost = (json.loads(cp_live.read_text(encoding="utf-8")) if cp_live.exists()
                    else json.loads(z.read(cp_n[0]).decode("utf-8")))
            out[E] = {
                "psi_eq": _read_parquet(eq_n[0]),
                "psi_dev": pd.concat([_read_parquet(n) for n in dv_n], ignore_index=True),
                "cost": cost,
                "n_shards": len(dv_n),
            }
    return out


def ridge_stats(eq: pd.DataFrame, dev: pd.DataFrame, is_B: bool | None = None) -> dict:
    """Collinearity of the two columns that carry omega and zeta in the DIFFERENCED design.

    is_B selects a single firm type (True=B, False=D); None pools both, which is what the
    reported ridge table uses. The per-type call exists for check_psi_matches_solve(), because
    cost_params stores its ridge fields per type.
    """
    m = dev.merge(eq, on=["firm", "is_B"], suffixes=("_d", "_e"))
    if is_B is not None:
        m = m[m["is_B"].to_numpy().astype(bool) == is_B]
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
    )


def load_cost_only() -> dict:
    """-> {routine: cost_params dict}, straight from cluster_processed, no psi involved.

    bbl_solve.py persists the design diagnostics it measured on ITS OWN psi
    (rbar_f, ridge_corr, ridge_cond, ridge_one_minus_R2, n_rows, n_firms). Everything both
    tables need is therefore already inside cost_params, and building from it alone makes a
    vintage mismatch impossible rather than merely detectable.
    """
    out = {}
    for E in ROUTINES:
        p = COST_DIR / f"cost_params_E{E}_spec_{SPEC}_{STAGE}.json"
        if p.exists():
            out[E] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"  [skip] E{E}: {p.name} not found")
    return out


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
        r"-(1+\zeta)\Delta\psi_4$ of \eqref{eq:17}."
    )
    return _wrap(body, col_fmt, r"$\psi_2/\psi_4$ Ridge",
                 "tab:bbl_ridge_diagnostic", header, foot, 9)


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
    the eq:17 criterion is kinked and potentially set-identified, so a normal approximation is
    not the right reference distribution and these should not be read as tests.
    """
    if est is None or se is None or not np.isfinite(est) or not np.isfinite(se) or se <= 0:
        return ""
    t = abs(est / se)
    return "***" if t >= 2.576 else "**" if t >= 1.96 else "*" if t >= 1.645 else ""


def build_cbar_panels(rows):
    r"""The identified combination and the health metric that governs whether it may be read.

    Split out of the parameter table deliberately: \bar c^\kappa is the only quantity this design
    identifies, and frac_bind is the gate on quoting it, so the two belong on one page together
    rather than buried under eight rows of individually meaningless coefficients.
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
    foot = (
        r"\textit{Notes:} $\bar c^\kappa = \omega^\kappa + \bar r^f\zeta^\kappa$ is the marginal "
        r"cost of deposits at the mean forward risk-free rate, and \emph{the only cost object "
        r"this design identifies}: $\Delta\psi_2$ and $\Delta\psi_4$ are collinear at "
        r"$\mathrm{corr}>0.9999$ (Table~\ref{tab:bbl_ridge_diagnostic}), so $\omega^\kappa$ and "
        r"$\zeta^\kappa$ are separately unidentified and are reported in "
        r"Table~\ref{tab:bbl_cost_identified} for completeness only. Quarterly units, and "
        r"excluding the $\boldsymbol{\gamma}'\boldsymbol{Z}$ shifters. "
        r"The standard error is a firm-block bootstrap SD (200 reps); the interval is "
        r"the subsampling $\sqrt{n}$-rate quantile CI ($b=n^{2/3}$ firms, 200 reps), which is the "
        r"appropriate route for a criterion that is kinked and potentially set-identified. "
        r"The last row is the share of the firm~$\times$~deviation revealed-preference "
        r"inequalities $g=V(\hat\sigma)-V(\tilde\sigma)\ge 0$ that \emph{fail} at "
        r"$\hat\theta$ --- deviations the fitted model says would have raised the firm's "
        r"value; a policy that is a genuine best response implies a share near $0$."
    )
    return _wrap(body, col_fmt,
                 r"BBL Identified Marginal Cost $\bar c^\kappa$ and Criterion Health",
                 "tab:bbl_cbar", header, foot, ncol)


def md_cbar_panels(rows):
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
    L += ["",
          "c-bar^kappa = omega^kappa + rbar_f * zeta^kappa, the only cost object this design "
          "identifies; omega and zeta separately are not (see the ridge table). SE is a "
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


def build_identified_panels(rows):
    r"""Estimates table in the paper's own layout: parameters down the rows, estimation routines
    across the columns as \ref{estimation:*} (rendered (III)-(VI)), and one panel per firm type.

    This replaces a routine x type row grid, which needed eight columns and had no room for
    gamma at all. Five columns fit the text block comfortably, standard errors sit under their
    estimates as in polfunc_k4.tex, and the type superscript moves out of every cell into the
    panel title -- V_Main's notation is omega^{\mathrm{B}}, zeta^{\mathrm{B}},
    (\boldsymbol{\gamma}^{\mathrm{B}})', and \bar c^\kappa for the identified combination.
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

    body = []
    for pi, (kappa, _lbl) in enumerate(BLOCKS):
        if not any((E, kappa) in by for E in Es):
            continue
        title = ("Brick-and-Mortar (B) Firms" if kappa == "B" else "Digital (D) Firms")
        if pi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Panel {'AB'[pi]}: {title}}}}} \\")
        body.append(r"\addlinespace[0.3ex]")
        body += line(rf"$\hat\omega^{{\mathrm{{{kappa}}}}}$",
                     lambda r: r["omega"], se_get=lambda r: r.get("omega_se"), stars=True)
        body += line(rf"$\hat\zeta^{{\mathrm{{{kappa}}}}}$",
                     lambda r: r["zeta"], fmt="{:.2f}",
                     se_get=lambda r: r.get("zeta_se"), se_fmt="{:.1f}", stars=True)
        body.append(r"\addlinespace[0.3ex]")
        # The gamma block is headed by the bare symbol, flush left like the other parameters;
        # the shifters it loads on are indented beneath it.
        body.append(rf"$\hat{{\boldsymbol{{\gamma}}}}^{{\mathrm{{{kappa}}}}}$"
                    + " & " * len(Es) + r" \\")
        for key, lab in Z_LABELS:
            body += line(r"\hspace{1em}" + lab,
                         (lambda k: lambda r: (r["gamma"] or {}).get(k))(key),
                         se_get=(lambda k: lambda r: (r["gamma_se"] or {}).get(k))(key),
                         stars=True)
        # Regression information, separated from the parameters as in the sleepiness tables.
        # (rbar^f is not a parameter of eq:8 — it lives with c-bar, which it defines.)
        body.append(r"\midrule")
        body += line(r"Firms", lambda r: r["n_firms"], fmt="{:.0f}")
        body += line(r"Inequalities $n$", lambda r: r["n"], fmt="{:,.0f}")

    foot = (
        r"\textit{Notes:} Deposit-servicing marginal cost parameters of \eqref{eq:8} from the "
        r"\textcite{bajari2007estimating} moment-inequality problem \eqref{eq:17}, spec.~12, "
        r"extended stage, estimated separately by firm type; columns are the estimation routines "
        r"of Section~\ref{sec:empirical:sleep}. Quarterly units; the cost-shifter ratios are in "
        r"percentage points and the Basel index a fraction, both lagged one quarter. Standard "
        r"errors in parentheses are firm-block bootstrap SDs (200 reps). "
        r"\emph{None of these parameters is separately identified}: $\Delta\psi_2$ and "
        r"$\Delta\psi_4$ are collinear at $\mathrm{corr}>0.9999$ "
        r"(Table~\ref{tab:bbl_ridge_diagnostic}), so the criterion is flat along "
        r"$\omega=-\bar r^f\zeta$ and $\hat\omega^\kappa$, $\hat\zeta^\kappa$ and "
        r"$\hat{\boldsymbol{\gamma}}^\kappa$ slide freely along that ridge --- one block returns "
        r"$\hat\omega>0$ with $\hat\zeta<0$ while the identified combination barely moves. The "
        r"table is reported for completeness; the estimand is $\bar c^\kappa$ in "
        r"Table~\ref{tab:bbl_cbar}. Significance markers use the normal approximation to the "
        r"bootstrap SD and are shown by convention only --- the criterion in \eqref{eq:17} is "
        r"kinked and potentially set-identified, so that reference distribution does not apply "
        r"and they should not be read as tests. *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )
    return _wrap(body, col_fmt,
                 r"BBL Deposit-Servicing Cost Parameters, by Firm Type",
                 "tab:bbl_cost_identified", header, foot, ncol)


def md_identified_panels(rows):
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
        row(f"omega^{kappa}", lambda r: f"{r['omega']:.3f}{_stars(r['omega'], r['omega_se'])} "
                                        f"({r['omega_se']:.3f})")
        row(f"zeta^{kappa}", lambda r: f"{r['zeta']:.2f}{_stars(r['zeta'], r['zeta_se'])} "
                                       f"({r['zeta_se']:.1f})")
        row(f"**gamma^{kappa}**", lambda r: "")
        for key, lab in Z_LABELS:
            row("  " + lab.replace("($t-1$)", "(t-1)"),
                (lambda k: lambda r: (
                    f"{(r['gamma'] or {}).get(k, float('nan')):.3f}"
                    f"{_stars((r['gamma'] or {}).get(k), (r['gamma_se'] or {}).get(k))} "
                    f"({(r['gamma_se'] or {}).get(k, float('nan')):.3f})"))(key))
        row("*Firms*", lambda r: f"{r['n_firms']}")
        row("*Inequalities n*", lambda r: f"{r['n']:,}")
    L += ["",
          "Columns are estimation routines E3/E4. SEs in parentheses are firm-block bootstrap SDs "
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
    # Only the two cost tables come from cost_params. The ridge table is per-routine over the
    # pooled design and is built from the psi archive by --from-psi; it is deliberately NOT
    # written here, so a cost-side rebuild cannot restyle or overwrite it.
    for name, txt in {"tab_bbl_cbar.tex": build_cbar_panels(rows),
                      "tab_bbl_cost_identified.tex": build_identified_panels(rows)}.items():
        for dest in (DRAFTS_DIR,):
            (dest / name).write_text(txt, encoding="utf-8")
        print(f"  wrote {name} -> Drafts/")
    for name, txt in {"tab_bbl_cbar.md": md_cbar_panels(rows),
                      "tab_bbl_cost_identified.md": md_identified_panels(rows)}.items():
        (DRAFTS_DIR / name).write_text(txt + "\n", encoding="utf-8")
        print(f"  wrote {name} -> Drafts/")
    print("\n\\input{tab_bbl_cbar.tex}            % tab:bbl_cbar   -- the estimand")
    print("\\input{tab_bbl_cost_identified.tex}  % tab:bbl_cost_identified -- parameters")
    print("\\input{tab_bbl_ridge_diagnostic.tex} % tab:bbl_ridge_diagnostic (--from-psi)")
    print("(needs booktabs + xltabular, both already in V_Main)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi-zip", default=str(PSI_ZIP))
    ap.add_argument("--from-psi", action="store_true",
                    help="recompute the ridge columns from the psi archive instead of reading "
                         "them out of cost_params. Requires the archive to BE the psi the solve "
                         "consumed; verified by check_psi_matches_solve().")
    ap.add_argument("--allow-vintage-mismatch", action="store_true",
                    help="with --from-psi: emit tables even if the psi archive is not the psi "
                         "the solve used. Only for inspecting a known-mixed pair; not reportable.")
    a = ap.parse_args()

    if not a.from_psi:
        return main_from_json()

    zp = pathlib.Path(a.psi_zip)
    if not zp.exists():
        raise SystemExit(f"psi archive not found: {zp}\nPass --psi-zip.")
    print(f"reading {zp.name} ({zp.stat().st_size/1e6:.1f} MB)")

    data = load_from_zip(zp)
    if not data:
        raise SystemExit("no routines recovered from the archive.")

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
        print(f"  E{E}: n={s['n']:,} ({s['n_shards']} shards)  corr={s['corr']:.8f}  "
              f"cond={s['cond']:,.0f}  rbar={s['rbar']:.5f}  1-R2={s['one_minus_r2']:.2e}  "
              f"across-D={s['across_delta_pct']:.1f}%")

    # The ridge table is the ONLY thing this path writes. It is per routine over the pooled
    # design, which needs the psi itself — cost_params stores its ridge fields per firm type, so
    # the pooled columns (Ratio, CV%, the shared #Delta) cannot be recovered from JSON. The two
    # cost tables come from cost_params via the default path and are left untouched here.
    for name, txt in {"tab_bbl_ridge_diagnostic.tex": build_ridge(ridge)}.items():
        for dest in (DRAFTS_DIR,):
            (dest / name).write_text(txt, encoding="utf-8")
        print(f"  wrote {name} -> Drafts/")
    for name, txt in {"tab_bbl_ridge_diagnostic.md": md_ridge(ridge)}.items():
        (DRAFTS_DIR / name).write_text(txt + "\n", encoding="utf-8")
        print(f"  wrote {name} -> Drafts/")
    print("\n\\input{tab_bbl_ridge_diagnostic.tex}  (needs booktabs + xltabular)")


if __name__ == "__main__":
    main()
