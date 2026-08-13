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

TABLES_DIR = rc.TABLES_DIR
DRAFTS_DIR = rc.DRAFTS_DIR
EST_OUT = rc.DATA_DIR / "ESTIMATION_OUTPUT"
PSI_ZIP = EST_OUT / "BBL_OUTPUT" / "cluster_raw" / "psi_cost.zip"
COST_DIR = EST_OUT / "BBL_OUTPUT" / "cluster_processed"

ROUTINES = (5, 6, 7, 8)
BLOCKS = (("B", r"Brick \& mortar"), ("D", "Digital"))
SPEC, STAGE = 12, "extended"

# A quarterly rate -> pp/yr uses the code's own x400 convention (estimation_bbl_2_fwd_sim.jl:473).
PP_YR = 400.0


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
            # predate the c_bar/ridge fields that estimation_bbl_3_solve.py now persists.
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


def ridge_stats(eq: pd.DataFrame, dev: pd.DataFrame) -> dict:
    """Collinearity of the two columns that carry omega and zeta in the DIFFERENCED design."""
    m = dev.merge(eq, on=["firm", "is_B"], suffixes=("_d", "_e"))
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


def build_identified(rows, ridge):
    # Firm type carries "Brick & mortar" and the 95% CI carries $[-0.0633,0.0390]$; both are
    # wider than an equal share of the text block, so both get fixed columns and the five
    # single-number columns split what is left as X. Sized against the widest realised cell in
    # each: shrinking the CI column below ~2.1cm puts the negative-lower-bound rows over the
    # margin, which is where the intervals that straddle zero live.
    col_fmt = (r">{\raggedright\arraybackslash}p{1.3cm} "
               r">{\raggedright\arraybackslash}p{1.5cm} "
               r"*{4}{>{\centering\arraybackslash}X} "
               r">{\centering\arraybackslash}p{2.2cm} "
               r">{\centering\arraybackslash}X")
    header = (r"Routine & Firm type & $\hat\omega$ (SE) & $\hat\zeta$ (SE) & $\bar r^f_q$ & "
              r"$\hat c\,(\bar r^f)$ (SE) & 95\% CI & pp/yr")
    body = []
    for i, r in enumerate(rows):
        first = (i == 0) or rows[i - 1]["E"] != r["E"]
        if first and i:
            body.append(r"\addlinespace[0.4ex]")
        ose = f" ({r['omega_se']:.3f})" if r.get("omega_se") else ""
        zse = f" ({r['zeta_se']:.1f})" if r.get("zeta_se") else ""
        cse = f" ({r['cbar_se']:.4f})" if r.get("cbar_se") else ""
        # \scriptsize and no thin space: the interval is the widest cell in the table and the
        # only one that cannot be shortened without dropping a significant digit.
        ci = (rf"{{\scriptsize $[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]$}}"
              if r.get("ci_lo") is not None else "---")
        body.append(" & ".join([
            rc.est_ref(r["E"]) if first else "",
            r["label"],
            f"${r['omega']:.3f}${ose}", f"${r['zeta']:.2f}${zse}", _f(r["rbar"], 5),
            r"\textbf{" + f"{r['cbar']:.4f}" + "}" + cse,
            ci,
            r"\textbf{" + f"{r['cbar'] * PP_YR:.2f}" + "}",
        ]) + r" \\")
    cmin = min(x["cbar"] for x in rows) * PP_YR
    cmax = max(x["cbar"] for x in rows) * PP_YR
    excl = sum(1 for x in rows if x.get("ci_lo") is not None and x["ci_lo"] > 0)
    # How much sharper is the combination than its own components?
    gain = max((x["omega_se"] / x["cbar_se"]) for x in rows
               if x.get("omega_se") and x.get("cbar_se"))
    wz = [x["wz_corr"] for x in rows if x.get("wz_corr") is not None]
    foot = (
        r"\textit{Notes:} Marginal cost of deposits from the \textcite{bajari2007estimating} "
        r"moment-inequality problem \eqref{eq:17}, spec.~12, extended stage, estimated "
        r"separately by firm type. "
        r"$\bar r^f_q$ is the $\beta^t$-weighted mean forward risk-free rate implied by the "
        r"design, $\bar r^f_q=\Delta\psi_4/\Delta\psi_2$. \emph{Only "
        r"$\hat c(\bar r^f)=\hat\omega+\bar r^f\hat\zeta$ is identified}, and it is positive in "
        rf"all eight blocks ({cmin:.1f}--{cmax:.1f} pp/yr before the $\gamma'Z$ shifters), with "
        rf"{excl} of {len(rows)} 95\% intervals excluding zero. Standard errors for "
        r"$\hat\omega,\hat\zeta,\hat c$ are firm-block bootstrap SDs (200 reps); the interval "
        r"for $\hat c$ is the subsampling $\sqrt{n}$-rate quantile CI ($b=n^{2/3}$ firms, 200 "
        r"reps), which is the appropriate route for a criterion that is kinked and potentially "
        r"set-identified. $\hat\omega$ and $\hat\zeta$ are reported only for completeness: they "
        r"are \emph{not} separately identified, because $\Delta\psi_2$ and $\Delta\psi_4$ are "
        r"collinear at $\mathrm{corr}>0.9999$ (Table~\ref{tab:bbl_ridge_diagnostic}). The "
        r"criterion is therefore flat along $\omega=-\bar r^f\zeta$, which is why $\hat\omega<0$ "
        r"throughout without implying negative marginal cost, and why the bootstrap draws of "
        rf"$(\hat\omega,\hat\zeta)$ correlate at {min(wz):.4f} to {max(wz):.4f}. The combination "
        rf"is up to {gain:.0f}$\times$ more precisely estimated than $\hat\omega$ alone. "
        r"\emph{Caveat:} all eight blocks trip the solver's own $\mathrm{frac\_bind}\approx"
        r"\tfrac12$ gate, indicating the $\pm$ deviation grid carries little identifying "
        r"content; these magnitudes should be treated as provisional until that is resolved."
    )
    return _wrap(body, col_fmt,
                 r"BBL Marginal Cost of Deposits: the Identified Combination (Spec.~12, Extended)",
                 "tab:bbl_cost_identified", header, foot, 8)


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


# ── markdown twins (for identification_notes.md) ─────────────────────────────
def md_identified(rows):
    L = ["| Routine | Firm type | omega-hat (SE) | zeta-hat (SE) | rbar_f | "
         "**c-hat(rbar_f)** (SE) | 95% CI | **pp/yr** |",
         "|---|---|---:|---:|---:|---:|:--:|---:|"]
    for r in rows:
        ose = f" ({r['omega_se']:.3f})" if r.get("omega_se") else ""
        zse = f" ({r['zeta_se']:.1f})" if r.get("zeta_se") else ""
        cse = f" ({r['cbar_se']:.4f})" if r.get("cbar_se") else ""
        ci = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if r.get("ci_lo") is not None else "--")
        L.append(f"| E{r['E']} | {r['label'].replace(chr(92)+'&','&')} | "
                 f"{r['omega']:.3f}{ose} | {r['zeta']:.2f}{zse} | {r['rbar']:.5f} | "
                 f"**{r['cbar']:.4f}**{cse} | {ci} | **{r['cbar']*PP_YR:.2f}** |")
    L += ["",
          "SEs are firm-block bootstrap SDs (200 reps); the CI for c-hat is the subsampling "
          "sqrt(n)-rate quantile interval (b = n^(2/3) firms, 200 reps). Only c-hat(rbar_f) is "
          "identified — omega-hat and zeta-hat are shown for completeness only."]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi-zip", default=str(PSI_ZIP))
    a = ap.parse_args()

    zp = pathlib.Path(a.psi_zip)
    if not zp.exists():
        raise SystemExit(f"psi archive not found: {zp}\nPass --psi-zip.")
    print(f"reading {zp.name} ({zp.stat().st_size/1e6:.1f} MB)")

    data = load_from_zip(zp)
    if not data:
        raise SystemExit("no routines recovered from the archive.")
    ridge, rows = collect(data)

    for E in sorted(ridge):
        s = ridge[E]
        print(f"  E{E}: n={s['n']:,} ({s['n_shards']} shards)  corr={s['corr']:.8f}  "
              f"cond={s['cond']:,.0f}  rbar={s['rbar']:.5f}  1-R2={s['one_minus_r2']:.2e}  "
              f"across-D={s['across_delta_pct']:.1f}%")

    tex = {
        "tab_bbl_cost_identified.tex": build_identified(rows, ridge),
        "tab_bbl_ridge_diagnostic.tex": build_ridge(ridge),
    }
    for name, txt in tex.items():
        for dest in (TABLES_DIR, DRAFTS_DIR):
            (dest / name).write_text(txt, encoding="utf-8")
        print(f"  wrote {name} -> Rout/ + Drafts/")

    for name, txt in {"tab_bbl_cost_identified.md": md_identified(rows),
                      "tab_bbl_ridge_diagnostic.md": md_ridge(ridge)}.items():
        (DRAFTS_DIR / name).write_text(txt + "\n", encoding="utf-8")
        print(f"  wrote {name} -> Drafts/")

    print("\n\\input{tab_bbl_cost_identified.tex} / \\input{tab_bbl_ridge_diagnostic.tex}"
          "  (needs booktabs)")


if __name__ == "__main__":
    main()
