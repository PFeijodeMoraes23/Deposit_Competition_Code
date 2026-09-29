"""
make_weakiv_table.py
====================
The appendix table of weak-instrument diagnostics for the deposit-price coefficient alpha,
tab_weakiv_spec12.tex (label tab:weakiv_spec12), one column per single-index routine.

  A. First-stage strength by deposit-type block: Montiel Olea-Pflueger effective F with its
     10% worst-case-bias critical value, and the Patnaik effective number of instruments.
  B. Robust sets on the sixteen-instrument design: Kleibergen LM, the tF interval, and the
     Anderson-Rubin set with Hansen's J (the AR minimum).
  C. Just-identified sets for the estimator the paper reports (the engine's own instrument),
     as estimated and cross-fitted.
  D. The engine's linear step run on each deposit-type block alone.

Every number is read from the battery JSONs; nothing is computed here:
  cluster_processed/weak_iv_ext1.json        blp_weak_iv.py --delta-stage ext1   (A, B)
  cluster_processed/diag_moment_reduction.json  blp_moment_reduction.py          (C, D)
Both are inverted against the structural delta of the reported (ext1) rung. The first-stage
block does not depend on delta, so panel A is the same whichever delta the battery used.

Writes to Drafts/Deposit Competition. Usage:  python make_weakiv_table.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import math
import sys

from utils import paths as _paths
from utils import routines as _routines

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CP = _paths.blp_results_dir() / "cluster_processed"
WEAK_IV_EXT1 = CP / "weak_iv_ext1.json"
DIAG_MR = CP / "diag_moment_reduction.json"
OUT = _paths.drafts_dir() / "tab_weakiv_spec12.tex"
ROUTINES = list(_routines.LINK_ESTS)
# Width of the notes row, and so of the table: longtable sizes itself from its widest row.
NOTES_WIDTH = r"0.82\linewidth"
BLOCKS = [("all", "All types (pooled)"), ("type12", "Types 1--2"), ("type45", "Types 4--5"),
          ("type4", "Type 4"), ("type5", "Type 5")]


def _ok(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _f(x, d):
    return f"{x:.{d}f}" if _ok(x) else "---"


def _s(x, d):
    """Number for math mode, where '-' is the minus sign."""
    return f"{x:.{d}f}" if _ok(x) else "---"


def _segments_tex(segs, grid_lo, grid_hi, d=2):
    """Union of accepted intervals; an endpoint on the search-grid edge is open at infinity. A
    union of three or more pieces is stacked on two lines so the column stays narrow."""
    if not segs:
        return r"$\emptyset$"
    parts = []
    for lo, hi in segs:
        a = r"(-\infty" if lo <= grid_lo + 1e-9 else f"[{lo:.{d}f}"
        b = r"\infty)" if hi >= grid_hi - 1e-9 else f"{hi:.{d}f}]"
        parts.append(f"{a},{b}")
    if len(parts) < 3:
        return "$" + r"\cup".join(parts) + "$"
    h = (len(parts) + 1) // 2
    top, bot = r"\cup".join(parts[:h]), r"\cup".join(parts[h:])
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}${top}\cup$\\${bot}$\end{{tabular}}"


def _sset_tex(s, grid_lo=-2.0, grid_hi=2.0, d=2):
    """A diag_moment_reduction set dict ({empty} or {lo, hi, disconnected}) as TeX."""
    if not s:
        return "---"
    if s.get("empty"):
        return r"$\emptyset$"
    txt = _segments_tex([[s["lo"], s["hi"]]], grid_lo, grid_hi, d)
    return txt[:-1] + r"^{\dagger}$" if s.get("disconnected") else txt


def _variant(diag, k, name):
    for r in diag:
        if r.get("routine") == k:
            for v in r.get("variants") or []:
                if v.get("variant") == name:
                    return v
    return {}


def _routine(diag, k):
    return next((r for r in diag if r.get("routine") == k), {})


def build(wiv, diag):
    ks = [k for k in ROUTINES if str(k) in wiv and _routine(diag, k)]
    if not ks:
        raise ValueError("tab_weakiv_spec12: the battery JSONs hold none of the routines "
                         f"{ROUTINES}; re-run blp_weakiv_job.sh for the current lineup.")
    nc = len(ks) + 1
    rows = []

    # Rows end in \\* (no page break after them): the table is half a page, and a longtable
    # split would strand the last rows and the notes on the next page.
    def row(label, cells):
        rows.append("    " + " & ".join([label] + cells) + r" \\*")

    def panel(title, first=False):
        if not first:
            rows.append(r"    \addlinespace[0.8ex]")
        rows.append(rf"    \multicolumn{{{nc}}}{{@{{}}l}}{{\textit{{{title}}}}} \\*")

    # A. first stage by block (delta-invariant: computed from the spread and Z alone)
    panel(r"A.\ First stage: effective $F$ [critical value]", first=True)
    for key, lab in BLOCKS:
        cells = []
        for k in ks:
            r = wiv[str(k)].get(key) or {}
            cv = (r.get("mop_cv") or {}).get("bias10")
            cells.append(f"${_f(r.get('effective_F'), 2)}$ $[{_f(cv, 1)}]$" if r else "---")
        row(r"\quad " + lab, cells)
    row(r"\quad Effective no.\ of instruments (of 16)",
        [f"${_f((wiv[str(k)].get('all') or {}).get('k_eff'), 1)}$" for k in ks])

    # B. sixteen-instrument robust sets, pooled sample
    panel(r"B.\ Sixteen instruments, all types: $95\%$ sets")
    cells_lm, cells_tf, cells_ar, cells_j = [], [], [], []
    for k in ks:
        r = wiv[str(k)].get("all") or {}
        g = r.get("grid") or [-2.0, 2.0]
        cells_lm.append(_segments_tex(r.get("lm_ci_segments"), g[0], g[1])
                        if r.get("lm_ci_segments") is not None else "---")
        tf = r.get("tf") or {}
        cells_tf.append(f"$[{_s(tf.get('tf_ci_low'), 2)},{_s(tf.get('tf_ci_high'), 2)}]$"
                        if _ok(tf.get("tf_ci_low")) else
                        ("undefined" if tf.get("tf_defined") is False else "---"))
        cells_ar.append(_segments_tex(r.get("ar_ci_segments") or [], g[0], g[1]))
        jp = r.get("hansen_J_p")
        pj = "{<}0.001" if _ok(jp) and jp < 0.001 else _f(jp, 3)
        cells_j.append(f"${_f(r.get('hansen_J'), 1)}$ $({pj})$")
    row(r"\quad Kleibergen LM", cells_lm)
    row(r"\quad tF", cells_tf)
    row(r"\quad Anderson--Rubin", cells_ar)
    row(r"\quad Hansen $J$, 15 d.f.\ ($p$-value)", cells_j)

    # C. just-identified sets for the reported estimator
    panel(r"C.\ Just-identified, the estimator's own instrument")
    ve = {k: _variant(diag, k, "ji_engine") for k in ks}
    vx = {k: _variant(diag, k, "ji_engine_split") for k in ks}
    row(r"\quad $\hat\alpha$ (SE)",
        [f"${_s(ve[k].get('alpha_2sls'), 3)}$ $({_f(ve[k].get('alpha_se'), 3)})$" for k in ks])
    row(r"\quad $95\%$ set, as estimated", [_sset_tex(ve[k].get("sset_wcb")) for k in ks])
    row(r"\quad $95\%$ set, cross-fitted", [_sset_tex(vx[k].get("sset_wcb")) for k in ks])

    # D. the linear step on each block
    panel(r"D.\ $\hat\alpha$ by deposit-type block [$95\%$ just-identified set]")
    for name, lab in (("ji_engine_t12", "Types 1--2, raw spread"),
                      ("ji_engine_t45", "Types 4--5, instrumented")):
        cells = []
        for k in ks:
            v = _variant(diag, k, name)
            cells.append(f"${_s(v.get('alpha_2sls'), 3)}$ {_sset_tex(v.get('sset_wcb'))}"
                         if v else "---")
        share = _variant(diag, ks[0], name).get("row_share")
        lab_full = lab + (f" ({100 * share:.0f}" + r"\%)" if _ok(share) else "")
        row(r"\quad " + lab_full, cells)

    r0 = _routine(diag, ks[0])
    gs = [(_routine(diag, k).get("n_clusters"), _routine(diag, k).get("gstar_used")) for k in ks]
    g_txt = ", ".join(str(g) for g, _ in gs)
    gstar_txt = _f(sum(x for _, x in gs if _ok(x)) / max(len(gs), 1), 1)
    B = r0.get("wcb_reps")
    notes = (
        r"\textit{Notes:} Spread in percentage points; $\delta$ at the reported "
        r"random-coefficients estimate. Conglomerate clusters "
        rf"($G={g_txt}$; $G^\ast={gstar_txt}$ effective). A: Montiel Olea--Pflueger effective "
        r"$F$ [simplified-test critical value, $10\%$ worst-case bias, $5\%$ size]; Patnaik "
        r"effective instrument count. B: spread instrumented on every row; "
        r"tF uses the Lee et al.\ (2022) critical value at the effective $F$. Anderson--Rubin "
        r"is empty because its minimum over $\alpha$ is Hansen's $J$, which rejects. CUE and "
        rf"Stock--Wright sets are not computable: with $G^\ast\approx{gstar_txt}$ the "
        r"$16\times16$ moment covariance is rank-deficient. LM, Anderson--Rubin and C--D sets "
        rf"invert a wild-cluster-bootstrap test (null imposed, Webb weights, {B} draws) over "
        r"$\alpha\in[-2,2]$, $\infty$ marking the grid edge; with one instrument this is the "
        r"Anderson--Rubin set. C--D: the instrument is the raw spread for types 1--2 and the "
        r"within-type first-stage fit for types 4--5 (cross-fitted: fit on the other half of "
        r"the clusters). The sign of $\hat\alpha$ rests on the exogeneity of the regulated types' "
        r"(1--2) spreads.")
    head = " & ".join([""] + [_routines.est_ref(k) for k in ks]) + r" \\"
    col = "l" + "c" * len(ks)
    lines = [
        r"\begin{spacing}{1.0}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{longtable}}[c]{{@{{\extracolsep{{\fill}}}}{col}@{{}}}}",
        r"    \caption{Weak-Instrument Diagnostics for the Deposit-Price Coefficient (Spec.~12)}",
        r"    \label{tab:weakiv_spec12} \\",
        r"    \toprule",
        "    " + head,
        r"    \midrule",
        r"    \endfirsthead",
        rf"    \multicolumn{{{nc}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        "    " + head,
        r"    \midrule",
        r"    \endhead",
        r"    \midrule",
        rf"    \multicolumn{{{nc}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        rf"    \multicolumn{{{nc}}}{{@{{}}p{{\dimexpr{NOTES_WIDTH}\relax}}@{{}}}}{{\footnotesize "
        + notes + r"} \\",
        r"    \endlastfoot",
        *rows,
        r"\end{longtable}",
        r"\end{spacing}",
    ]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--out", default=str(OUT), help=f"output .tex (default {OUT})")
    args = ap.parse_args()
    for p in (WEAK_IV_EXT1, DIAG_MR):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run blp_weakiv_job.sh (or the scripts it "
                                    f"calls) first.")
    wiv = json.load(open(WEAK_IV_EXT1))
    diag = json.load(open(DIAG_MR))
    tex = build(wiv, diag)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
