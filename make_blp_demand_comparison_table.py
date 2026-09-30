"""
make_blp_demand_comparison_table.py
=====================================
Cross-estimator comparison of an RC-BLP model across the sleepiness estimation strategies E3/E4,
Spec 12. One column per strategy (\\ref{estimation:*}); rows are the mean-utility coefficients θ₁
(Panel A) and the random-coefficient parameters θ₂ (Panel B). This is the RC analog of the logit
cross-estimator comparison (review §1.1): it puts the models side by side so the reader sees how the
estimates move across strategies.

TWO stages are emitted (see STAGE_SPECS):
  * ext1  (HEADLINE, no filename suffix) — a random coefficient on the deposit spread, its
    interactions with market demographics, and the ln-assets x income interaction. This is the last
    rung of the ladder reported per routine, so the cross-routine comparison ends where the
    within-routine tables (blp_rc_E*, blp_compare_E*) end. It is the headline because rc4 leaves the
    price coefficient near zero (partly wrong-signed), while ext1 delivers correctly-signed,
    near-unit-elastic estimates. Files: blp_demand_comparison{_noseg}_spec12.tex.
  * extended  (_full suffix) — the Full model with every freed random coefficient / interaction,
    kept as a robustness/appendix table. Files: blp_demand_comparison_full{_noseg}_spec12.tex.

The per-routine tables (blp_rc_E*, blp_compare_E*) run the ladder through ext1 either way. ext2 and
extended are the degenerate rungs (SEs inflate ~6-10x on a flat objective), which is why the ladder
is cut at ext1 there; `full` (the old 5-RC rung) is not a stage at all — see
make_blp_rc_table.STAGES / RC_TABLE_STAGES, the single source of truth for the ladder.

Reads  blp_results_E{3,4}_spec_12_{stage}{engine_suffix}.json.
Reuses the label maps + formatting (WCB-p/t(G*) stars, on-bound σ dagger) and the shared note
wording (demand_note, from config/table_notes.toml) from make_blp_rc_table.

Usage
-----
  python make_blp_demand_comparison_table.py                 # E3/E4, IFT engine, BOTH stages
  python make_blp_demand_comparison_table.py --routines 3
  python make_blp_demand_comparison_table.py --engine numerical
  python make_blp_demand_comparison_table.py --stages ext1    # only the ext1 (headline) table

Output (× {with-seg, _noseg})
------
  ext1:  Drafts/  blp_demand_comparison{_noseg}_spec12{engine}.tex
  full:  Drafts/  blp_demand_comparison_full{_noseg}_spec12{engine}.tex
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import math

import make_blp_rc_table as rc   # label maps, decode_theta2, fmt_coef, demand_note, est_ref, loaders
from utils import routines as _routines
import sys

# Windows consoles default to cp1252 and raise UnicodeEncodeError on any non-ASCII
# character in a print (phi, arrows, Upsilon, x). That usually fires on a STATUS line after
# the real work is done, so the script exits non-zero and reports failure for a computation
# that succeeded -- three such false failures on 2026-07-29. Force UTF-8 (no-op if already).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Display covers the whole reported lineup. link_ests names the routines that share the
# single-index link and drives the cluster run set; which routines a TABLE shows is a
# separate choice, so these read `active` instead.
DEFAULT_ESTS = list(_routines.LINK_ESTS)

# Which RC stage each comparison table reports.
#   stage        : checkpoint stage read from disk
#   file_lbl     : filename/label infix ("" => the default no-subscript table; "_full" => the Full one)
#   caption_tail : appended to the "BLP Demand Estimation" caption
# (What each column IS is described in V_Main's prose next to the \input, not in the note.)
STAGE_SPECS = {
    "ext1": dict(stage="ext1", file_lbl="", caption_tail=""),
    "full": dict(stage="extended", file_lbl="_full", caption_tail=r" --- Full Specification"),
}


def build_table(ests, suffix: str = "", show_segments: bool = True,
                stage: str = "extended", file_lbl: str = "",
                caption_tail: str = "") -> str:
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
    # One SE sentence must describe every column. A mix of se_method across routines means the
    # checkpoints on disk are from different vintages/runs — publishing one sentence over mixed
    # columns misdescribes at least one of them, so refuse instead.
    _methods = {e: (data[e] or {}).get("se_method", "none") for e in avail}
    if len(set(_methods.values())) > 1:
        raise SystemExit(f"[demand-comparison] mixed se_method across routines {_methods} — "
                         "stale checkpoint vintage; re-run/ingest the missing routine first.")
    lbl_suffix = "" if show_segments else "_noseg"
    # The note is the wording shared with the logit comparison (config/table_notes.toml, read by
    # blp_logit.jl too). The headline table (`rc_comparison`) leaves the omitted segment dummies to
    # V_Main's text, which states them before the \input; the _full appendix table (`rc_full`) has
    # no such sentence and says it in the note. The boundary clause appears only when some column
    # has a Σ on the bound (the dagger rows), whose other SEs are then conditional on it
    # (blp_se_common.jl gmm_cluster_ses profiles it out of the covariance).
    skip = ["segments"] if show_segments else []
    if not rc.sigma_on_bound(data[e] for e in avail):
        skip.append("boundary")
    note_body = rc.demand_note("rc_full" if file_lbl else "rc_comparison",
                               se_method=_methods[avail[0]], skip=skip)

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
        + rc.note_cell(note_body) +
        r"} \\",
        r"    \endlastfoot",
        "",
    ]

    # ── Panel A: θ₁ ───────────────────────────────────────────────────────────
    # Body rows end in \\* (not \\): the table fits on one printed page, and \\* forbids a
    # page break after that row so longtable/xltabular moves the whole block together instead
    # of splitting it (the \endfoot "continued" machinery is still there for the rare overflow).
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel A: Mean Utility ($\theta_1$)}} \\*")
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
        lines.append("    " + " & ".join(cvals) + r" \\*")
        if any(s.strip() for s in svals[1:]):
            lines.append("    " + " & ".join(svals) + r" \\*")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Panel B: θ₂ ───────────────────────────────────────────────────────────
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel B: Random Coefficients ($\theta_2$)}} \\*")
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
        lines.append("    " + " & ".join(cvals) + r" \\*")
        if any(s.strip() for s in svals[1:]):
            lines.append("    " + " & ".join(svals) + r" \\*")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Footer statistics ─────────────────────────────────────────────────────
    lines.append(r"    \midrule")
    def stat(name, fn):
        return "    " + name + " & " + " & ".join(fn(e) for e in avail) + r" \\*"
    def q_of(e):
        q = data[e].get("Q_value");  return f"${rc.fmt3(q)}$" if q is not None else "---"
    def n_of(e):
        n = data[e].get("n_obs");    return f"{n:,}" if n is not None else "---"
    def g_of(e):
        g = data[e].get("G_star");   return rc.fmt3(g) if g is not None else "---"
    def se_of(e):
        return rc.semi_elast_cell(data[e], e)         # α̂·mean(ρ(1−s)), average-market plug-in
    # Same labels and order as the logit comparison footer (blp_logit.jl), so the two demand
    # tables read row for row.
    lines += [
        stat(rc.ELAST_ROW_LABEL, se_of),
        stat(r"Observations", n_of),
        stat(rc.Q_ROW_LABEL, q_of),
        stat(rc.GSTAR_ROW_LABEL, g_of),
        r"\end{xltabular}",
        r"\end{spacing}",
    ]
    return "\n".join(lines)


def main():
    _across = "/".join(f"E{e}" for e in DEFAULT_ESTS)
    ap = argparse.ArgumentParser(description=f"RC-BLP cross-estimator comparison across "
                                             f"{_across} (ext1 headline + Full)")
    ap.add_argument("--routines", default=_routines.csv(DEFAULT_ESTS))
    ap.add_argument("--engine", choices=["ift", "numerical", "cue"], default="ift",
                    help="Engine whose results to read (ift→un-suffixed, numerical→_num, cue→_cue). "
                         "The suffix also lands in the output filename, so a non-ift engine writes a "
                         "NEW .tex beside the headline.")
    ap.add_argument("--stages", default="ext1,full",
                    help="comma list of STAGE_SPECS keys to emit (default: both)")
    args = ap.parse_args()
    suffix = rc.ENGINE_SUFFIX[args.engine]
    ests   = [int(x) for x in args.routines.split(",") if x.strip()]
    want   = [s.strip() for s in args.stages.split(",") if s.strip()]

    for key in want:
        spec = STAGE_SPECS.get(key)
        if spec is None:
            print(f"[demand-comparison] unknown stage '{key}' (have {list(STAGE_SPECS)}) — skipped")
            continue
        # ONLY the no-segment variant is emitted. The with-segments twin was retired: the segment
        # dummies it added duplicated four rows and no information — and kept the appendix long.
        # The footnote states they are included but not reported (same sentence as the logit
        # comparison note). Pass show_segments=True to build_table if a referee ever wants the
        # with-segments layout back.
        for show_seg, seg_lbl in ((False, "_noseg"),):
            tex = build_table(ests, suffix, show_segments=show_seg,
                              stage=spec["stage"], file_lbl=spec["file_lbl"],
                              caption_tail=spec["caption_tail"])
            if not tex:
                continue
            fname = f"blp_demand_comparison{spec['file_lbl']}{seg_lbl}_spec12{suffix}.tex"
            for dest in (rc.DRAFTS_DIR,):
                dest.mkdir(parents=True, exist_ok=True)
                (dest / fname).write_text(tex, encoding="utf-8")
                print(f"  saved [{key}]: {dest / fname}")
    print("\nDone. \\input needs \\usepackage{longtable,booktabs,setspace}.")


if __name__ == "__main__":
    main()
