"""
make_blp_demand_comparison_table.py
=====================================
Cross-estimator comparison of an RC-BLP model across the sleepiness estimation strategies E5-E8,
Spec 12. One column per strategy (\\ref{estimation:*}); rows are the mean-utility coefficients θ₁
(Panel A) and the random-coefficient parameters θ₂ (Panel B). This is the RC analog of the logit
cross-estimator comparison (review §1.1): it puts the models side by side so the reader sees how the
estimates move across strategies.

TWO stages are emitted (see STAGE_SPECS):
  * rc4  (DEFAULT, no filename suffix) — the price-heterogeneity model: a random coefficient on the
    deposit spread plus its interactions with market demographics. This is the last well-identified
    stage; the fgc/asset interactions freed in ext1/ext2/extended are near-collinear with the spread
    interactions, so they leave α essentially unchanged while inflating its SE ~8x. rc4 is therefore
    the headline. Files: blp_demand_comparison{_noseg}_spec12.tex.
  * extended  (_full suffix) — the Full model with every freed random coefficient / interaction,
    kept as a robustness/appendix table. Files: blp_demand_comparison_full{_noseg}_spec12.tex.

Reads  blp_results_E{5,6,7,8}_spec_12_{stage}{engine_suffix}.json.
Reuses the label maps + formatting (t(G*) stars, on-bound σ dagger, se_note) from make_blp_rc_table.

Usage
-----
  python make_blp_demand_comparison_table.py                 # E5-E8, IFT engine, BOTH stages
  python make_blp_demand_comparison_table.py --routines 5,6,7,8
  python make_blp_demand_comparison_table.py --engine numerical
  python make_blp_demand_comparison_table.py --stages rc4     # only the rc4 (default) tables

Output (× {with-seg, _noseg})
------
  rc4 :  BLP_RESULTS/Rout/ + Drafts/  blp_demand_comparison{_noseg}_spec12{engine}.tex
  full:  BLP_RESULTS/Rout/ + Drafts/  blp_demand_comparison_full{_noseg}_spec12{engine}.tex
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import math

import make_blp_rc_table as rc   # label maps, decode_theta2, fmt_coef, se_note, est_ref, loaders

DEFAULT_ESTS = [5, 6, 7, 8]

# Which RC stage each comparison table reports.
#   stage        : checkpoint stage read from disk
#   file_lbl     : filename/label infix ("" => the default no-subscript table; "_full" => the Full one)
#   caption_tail : appended to the "BLP Demand Estimation" caption
#   blurb        : the "Each column is <blurb> ..." phrase in the notes
STAGE_SPECS = {
    "rc4": dict(
        stage="rc4", file_lbl="", caption_tail="",
        blurb=(r"the random-coefficients model with a random coefficient on the deposit spread and "
               r"its interactions with market demographics (the price-heterogeneity specification, "
               r"the last stage in which $\hat\alpha$ is precisely estimated)"),
    ),
    "full": dict(
        stage="extended", file_lbl="_full", caption_tail=r" --- Full Specification",
        blurb=(r"the Full random-coefficients model, freeing every random coefficient and "
               r"demographic interaction"),
    ),
}


def build_table(ests, suffix: str = "", show_segments: bool = True,
                stage: str = "extended", file_lbl: str = "",
                caption_tail: str = "", blurb: str = "") -> str:
    data  = {e: rc.load_stage(e, stage, suffix) for e in ests}
    data  = {e: d for e, d in data.items() if d is not None}
    avail = [e for e in ests if e in data]
    if not avail:
        print(f"[demand-comparison] no {stage} results found for {ests}")
        return ""

    ncols   = len(avail)
    # The no-segment variant spans the FULL \textwidth (it is the one pulled into the paper body);
    # the with-segments variant stays narrower and centered via \LTleft/\LTright.
    TABLE_W = r"\textwidth" if not show_segments else r"0.86\textwidth"
    hdr     = " & ".join(rc.est_ref(e) for e in avail)        # bare \ref{estimation:*} column heads
    G_map   = {e: data[e].get("G_star") for e in avail}
    rep     = data[avail[0]]
    sem     = rc.se_note(data.get(avail[-1]) or rep)
    lbl_suffix = "" if show_segments else "_noseg"
    # The segment dummies are nuisance controls whose coefficients are already printed IN FULL, per
    # routine, by the appendix tables blp_rc_E{5..8}_spec12.tex. So we omit them here and point the
    # reader at those tables rather than carrying a duplicate with-segments variant of this table
    # (which added four rows and no information). Never write "available on request" — they ARE
    # reported, just elsewhere.
    seg_note = "" if show_segments else (
        r" Segment dummies (S2--S5) are included in every strategy but not reported; the full "
        r"coefficient vector, including the segment dummies, appears in the per-routine tables "
        r"\ref{tab:blp_rc_est5_spec12}--\ref{tab:blp_rc_est8_spec12}.")

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
        rf"    \caption{{BLP Demand Estimation{caption_tail}}}",
        rf"    \label{{tab:blp_demand_comparison{file_lbl}{lbl_suffix}_spec12}} \\",
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
        rf"\scriptsize \textit{{Notes:}} Each column is {blurb} for one "
        rf"sleepiness estimation strategy, enumerated in Section~\ref{{sec:empirical:sleep}}.{seg_note} {sem}. "
        r"Significance from a "
        r"Student-$t$ reference with $G^*$ effective clusters: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
        r"The $\Sigma$'s are bounded $\Sigma\ge0$, and a $\dagger$ marks a $\Sigma$ at "
        r"the boundary ($\widehat{\Sigma}\approx0$), reported on the bound with no two-sided standard "
        r"errors \parencite{andrews1999}. $Q$: GMM overidentification statistic. Mean own-price elasticity is "
        r"the average-market plug-in $\hat\alpha\cdot\overline{\rho(1-s)}$ (unit-free, since the spread enters "
        r"in levels; representative-agent; the "
        r"exact RC value integrates the individual price coefficients). Spread in percentage points."
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
        is_sigma = lbl.startswith(r"$\Sigma$")
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
    def se_of(e):
        return rc.semi_elast_cell(data[e], e)         # α̂·mean(ρ(1−s)), average-market plug-in
    lines += [
        stat(r"Mean own-price elast.", se_of),
        stat(r"$Q$ (GMM)", q_of),
        stat(r"Observations", n_of),
        stat(r"Eff.\ Clusters ($G^*$)", g_of),
        r"\end{xltabular}",
        r"\end{spacing}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="RC-BLP cross-estimator comparison across E5-E8 "
                                             "(rc4 headline + Full)")
    ap.add_argument("--routines", default="5,6,7,8")
    ap.add_argument("--engine", choices=["ift", "numerical"], default="ift")
    ap.add_argument("--stages", default="rc4,full",
                    help="comma list of STAGE_SPECS keys to emit (default: both)")
    args = ap.parse_args()
    suffix = "_num" if args.engine == "numerical" else ""
    ests   = [int(x) for x in args.routines.split(",") if x.strip()]
    want   = [s.strip() for s in args.stages.split(",") if s.strip()]

    for key in want:
        spec = STAGE_SPECS.get(key)
        if spec is None:
            print(f"[demand-comparison] unknown stage '{key}' (have {list(STAGE_SPECS)}) — skipped")
            continue
        # ONLY the no-segment variant is emitted. The with-segments twin was retired: the segment
        # dummies it added are already printed per routine by blp_rc_E{5..8}_spec12.tex (both are in
        # the appendix), so it duplicated four rows and no information — and kept the appendix long.
        # The footnote now cross-references those tables. Pass show_segments=True to build_table if a
        # referee ever wants the with-segments layout back.
        for show_seg, seg_lbl in ((False, "_noseg"),):
            tex = build_table(ests, suffix, show_segments=show_seg,
                              stage=spec["stage"], file_lbl=spec["file_lbl"],
                              caption_tail=spec["caption_tail"], blurb=spec["blurb"])
            if not tex:
                continue
            fname = f"blp_demand_comparison{spec['file_lbl']}{seg_lbl}_spec12{suffix}.tex"
            for dest in (rc.TABLES_DIR, rc.DRAFTS_DIR):
                dest.mkdir(parents=True, exist_ok=True)
                (dest / fname).write_text(tex, encoding="utf-8")
                print(f"  saved [{key}]: {dest / fname}")
    print("\nDone. \\input needs \\usepackage{longtable,booktabs,setspace}.")


if __name__ == "__main__":
    main()
