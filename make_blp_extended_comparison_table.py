"""
make_blp_extended_comparison_table.py
=====================================
Cross-estimator comparison of the FULL (Extended, all-8-random-coefficient) RC-BLP model across the
sleepiness estimation strategies E5-E8, Spec 12. One column per strategy (\\ref{estimation:*});
rows are the mean-utility coefficients θ₁ (Panel A) and the random-coefficient parameters θ₂
(Panel B). This is the RC analog of the logit cross-estimator comparison (review §1.1): it puts the
headline full models side by side so the reader sees how the estimates move across strategies.

Reads  blp_results_E{5,6,7,8}_spec_12_extended{suffix}.json  (the last/Extended stage).
Reuses the label maps + formatting (t(G*) stars, on-bound σ dagger, se_note) from make_blp_rc_table.

Usage
-----
  python make_blp_extended_comparison_table.py                 # E5-E8, IFT engine
  python make_blp_extended_comparison_table.py --routines 5,6,7,8
  python make_blp_extended_comparison_table.py --engine numerical

Output
------
  BLP_RESULTS/Rout/blp_extended_comparison_spec12{suffix}.tex
  Drafts/Deposit Competition/blp_extended_comparison_spec12{suffix}.tex
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import math

import make_blp_rc_table as rc   # label maps, decode_theta2, fmt_coef, se_note, est_ref, loaders

DEFAULT_ESTS = [5, 6, 7, 8]
STAGE = "extended"   # the full model = all eight random coefficients freed


def build_table(ests, suffix: str = "", show_segments: bool = True) -> str:
    data  = {e: rc.load_stage(e, STAGE, suffix) for e in ests}
    data  = {e: d for e, d in data.items() if d is not None}
    avail = [e for e in ests if e in data]
    if not avail:
        print(f"[extended-comparison] no {STAGE} results found for {ests}")
        return ""

    ncols   = len(avail)
    TABLE_W = r"0.86\textwidth"      # narrower than \textwidth, centered via \LTleft/\LTright
    hdr     = " & ".join(rc.est_ref(e) for e in avail)        # bare \ref{estimation:*} column heads
    G_map   = {e: data[e].get("G_star") for e in avail}
    rep     = data[avail[0]]
    sem     = rc.se_note(data.get(avail[-1]) or rep)
    lbl_suffix = "" if show_segments else "_noseg"
    seg_note   = "" if show_segments else (r" Segment dummies (S2--S5) are included in every "
                                           r"strategy but not reported.")

    theta1_params = rep.get("param_names_theta1") or (["alpha"] + rc.X_COLS)
    if not show_segments:
        theta1_params = [p for p in theta1_params if not p.startswith("seg_")]

    lines = [
        r"\begin{spacing}{1.0}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTleft}{\fill}",     # center the (narrower) fixed-width table
        r"\setlength{\LTright}{\fill}",
        rf"\begin{{xltabular}}{{{TABLE_W}}}{{>{{\raggedright\arraybackslash}}p{{4.9cm}} "
        rf"*{{{ncols}}}{{>{{\centering\arraybackslash}}X}}}}",
        r"    \caption{BLP Demand Estimation}",
        rf"    \label{{tab:blp_extended_comparison{lbl_suffix}_spec12}} \\",
        r"    \toprule",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols + 1) + r"}{@{}p{\dimexpr" + TABLE_W + r"-2\tabcolsep\relax}@{}}{"
        r"\scriptsize \textit{Notes:} Each column is the \emph{full} random-coefficients model (all "
        r"eight random coefficients freed --- the `Extended' stage) for one sleepiness estimation "
        rf"strategy, enumerated in Section~\ref{{sec:empirical:sleep}}.{seg_note} {sem}. "
        r"\tiny Significance from a "   # everything from here is a step smaller (user request)
        r"Student-$t$ reference with $G^*$ effective clusters: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
        r"$\theta_1$: mean-utility coefficients (linear IV; demographics centered "
        r"$\tilde D=(D-\bar D)/\sigma$, so $\theta_1$ is the average-market coefficient). "
        r"$\theta_2$: random-coefficient parameters ($\sigma$ = std.\ dev., $\pi$ = demographic "
        r"interaction); the $\sigma$'s are bounded $\sigma\ge0$, and a $\dagger$ marks a $\sigma$ at "
        r"the boundary ($\hat\sigma\approx0$), reported on the bound with no two-sided SE "
        r"(Andrews 1999). $Q$: GMM overidentification statistic. Spread in percentage points."
        r"} \\",
        r"    \endlastfoot",
        "",
    ]

    # ── Panel A: θ₁ ───────────────────────────────────────────────────────────
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel A: Mean Utility ($\theta_1$)}} \\")
    lines.append(r"    \addlinespace[0.3ex]")
    for p in theta1_params:
        lbl = rc.THETA1_LABELS.get(p, p.replace("_", r"\_"))
        cvals, svals = [lbl], [""]
        for e in avail:
            d      = data[e]
            names  = d.get("param_names_theta1") or d.get("param_names", [])
            t1     = d.get("theta1", [])
            t1se   = d.get("theta1_se", [])
            t1pv   = d.get("theta1_pval", [])
            if p in names:
                i = names.index(p)
                v  = t1[i]   if i < len(t1)   else float("nan")
                se = t1se[i] if i < len(t1se) else 0.0
                pv = t1pv[i] if i < len(t1pv) else None
                c, s_ = rc.fmt_coef(v, se, G_map[e], pv)
            else:
                c, s_ = "-", ""
            cvals.append(c); svals.append(s_)
        lines.append("    " + " & ".join(cvals) + r" \\")
        if any(s.strip() for s in svals[1:]):
            lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Panel B: θ₂ ───────────────────────────────────────────────────────────
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel B: Random Coefficients ($\theta_2$)}} \\")
    lines.append(r"    \addlinespace[0.3ex]")

    # union of θ₂ labels across strategies, first-appearance order
    labels = []
    for e in avail:
        for lbl, *_ in rc.decode_theta2(data[e]):
            if lbl not in labels:
                labels.append(lbl)

    for lbl in labels:
        is_sigma = lbl.startswith(r"$\sigma$")
        cvals, svals = [lbl], [""]
        for e in avail:
            decoded = {l: (v, se, pv) for l, v, se, pv in rc.decode_theta2(data[e])}
            if lbl in decoded:
                v, se, pv = decoded[lbl]
                ob = is_sigma and v is not None and not (isinstance(v, float) and math.isnan(v)) \
                    and abs(v) < rc.SIGMA_BOUND_TOL
                c, s_ = rc.fmt_coef(v, se, G_map[e], pv, on_bound=ob)
            else:
                c, s_ = "-", ""
            cvals.append(c); svals.append(s_)
        lines.append("    " + " & ".join(cvals) + r" \\")
        if any(s.strip() for s in svals[1:]):
            lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Footer statistics ─────────────────────────────────────────────────────
    lines.append(r"    \midrule")
    def stat(name, fn):
        return "    " + name + " & " + " & ".join(fn(e) for e in avail) + r" \\"
    def q_of(e):
        q = data[e].get("Q_value");  return f"${q:.4f}$" if q is not None else "---"
    def n_of(e):
        n = data[e].get("n_obs");    return f"{n:,}" if n is not None else "---"
    def g_of(e):
        g = data[e].get("G_star");   return f"{g:.2f}" if g is not None else "---"
    lines += [
        stat(r"$Q$ (GMM)", q_of),
        stat(r"Observations", n_of),
        stat(r"Eff.\ Clusters ($G^*$)", g_of),
        r"\end{xltabular}",
        r"\end{spacing}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Full-model (Extended) RC-BLP comparison across E5-E8")
    ap.add_argument("--routines", default="5,6,7,8")
    ap.add_argument("--engine", choices=["ift", "numerical"], default="ift")
    args = ap.parse_args()
    suffix = "_num" if args.engine == "numerical" else ""
    ests   = [int(x) for x in args.routines.split(",") if x.strip()]

    # Two versions: with segment dummies (S2-S5) and without (they are nuisance controls).
    for show_seg, lbl in ((True, ""), (False, "_noseg")):
        tex = build_table(ests, suffix, show_segments=show_seg)
        if not tex:
            continue
        fname = f"blp_extended_comparison{lbl}_spec12{suffix}.tex"
        for dest in (rc.TABLES_DIR, rc.DRAFTS_DIR):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / fname).write_text(tex, encoding="utf-8")
            print(f"  saved: {dest / fname}")
    print("\nDone. \\input needs \\usepackage{longtable,booktabs,setspace}.")


if __name__ == "__main__":
    main()
