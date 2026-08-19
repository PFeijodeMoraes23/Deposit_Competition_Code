"""
make_blp_compare_table.py
=========================
One LANDSCAPE LaTeX table per routine that compares, side by side:
    Logit  |  sigma | rc2 | rc3 | rc4 | full | ext1 | ext2 | extended
i.e. the non-RC logit first, then each increasingly-complex RC-BLP version we ran on
the cluster (1 -> 8 freed random coefficients). Spec 12.

Data sources (post-2026-06-25 BLP_RESULTS reorg):
  * RC stages : cluster_raw/blp_results_E{est}_spec_12_{stage}.json   (IFT engine; θ₁+SE,
                θ₂ point estimates — θ₂ SEs are not computed so they print without stars)
  * Logit     : logit/logit_summary_spec_12.json  (entry "E{est}_full": θ₁ + clustered SE)

Reuses the label maps + formatting helpers from make_blp_rc_table.py.

Usage
-----
  python make_blp_compare_table.py            # E3/E4 (the cluster default set)
  python make_blp_compare_table.py --est 6
  python make_blp_compare_table.py --routines 5,6,7,8

Output
------
  BLP_RESULTS/.../Rout/blp_compare_E{est}_spec12.tex
  Drafts/Deposit Competition/blp_compare_E{est}_spec12.tex
  (wrap the document body in \\usepackage{pdflscape} + \\usepackage{longtable,booktabs,setspace})
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import math
import pathlib

import scipy.stats as stats
import make_blp_rc_table as rc   # label maps, decode_theta2, sigma_label, pi_label, STAGES
import sys

# Windows consoles default to cp1252 and raise UnicodeEncodeError on any non-ASCII
# character in a print (phi, arrows, Upsilon, x). That usually fires on a STATUS line after
# the real work is done, so the script exits non-zero and reports failure for a computation
# that succeeded -- three such false failures on 2026-07-29. Force UTF-8 (no-op if already).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RES_DIR    = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
RAW_DIR    = RES_DIR / "cluster_raw"
LOGIT_DIR  = RES_DIR / "logit"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = rc.DRAFTS_DIR
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Stop at ext1, matching the per-routine RC tables (rc.RC_TABLE_STAGES): ext2/extended are the
# degenerate rungs where the flat objective inflates every SE ~6-10x, so those columns carry no
# usable information. The Full model still appears in the _full demand-comparison table.
STAGES = rc.RC_TABLE_STAGES
# RC-stage headers by number of freed random coefficients (1→7). `full` is not a stage (dropped from
# rc.STAGES; σ(ln assets) removed → full≡rc4), so ext1=5 RC, ext2=6 RC.
STAGE_HEAD = {"sigma": "1 RC", "rc2": "2 RC", "rc3": "3 RC", "rc4": "4 RC",
              "ext1": "5 RC", "ext2": "6 RC", "extended": "Full"}


# ── loaders ───────────────────────────────────────────────────────────────────
def load_stage(est: int, stage: str) -> dict | None:
    p = RAW_DIR / f"blp_results_E{est}_spec_12_{stage}.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
        return d or None
    except (json.JSONDecodeError, ValueError):
        return None


def load_logit_full(est: int) -> dict | None:
    """Logit comparator: the 'full' sub-model (α + X_COLS, matches RC θ₁)."""
    p = LOGIT_DIR / "logit_summary_spec_12.json"
    if not p.exists():
        return None
    summ = json.load(open(p))
    for sm in ("full", "full_dtype", "core", "priceonly"):
        e = summ.get(f"E{est}_{sm}")
        if e:
            return e
    return None


# ── θ₁ cell from a logit entry (has se+tstat, not theta1_se) ──────────────────
def logit_cell(entry: dict, pname: str):
    names = entry.get("param_names", [])
    if pname not in names:
        return "-", ""
    i  = names.index(pname)
    v  = entry["theta1"][i]
    se = entry["se"][i]
    G  = entry.get("G_star")
    return rc.fmt_coef(v, se, G)


# ── table builder ─────────────────────────────────────────────────────────────
def build_table(est: int) -> str:
    stage_data = {s: load_stage(est, s) for s in STAGES}
    stage_data = {s: d for s, d in stage_data.items() if d is not None}
    avail = [s for s in STAGES if s in stage_data]
    if not avail:
        print(f"[E{est}] no RC stage results in {RAW_DIR}")
        return ""
    logit = load_logit_full(est)
    rc_sem = rc.se_note(stage_data.get("extended"))       # RC SE-method sentence (method-aware)

    cols = (["logit"] if logit else []) + avail          # column keys, left→right
    ncols = len(cols)
    head = []
    for c in cols:
        head.append("Logit" if c == "logit" else STAGE_HEAD[c])
    hdr_cols = " & ".join(head)
    col_fmt = rc.equal_col_fmt(ncols, "4.6cm")   # equal-width columns fill the line (no bulging column)

    rep = stage_data[avail[0]]
    theta1_params = rep.get("param_names_theta1") or (["alpha"] + rc.X_COLS)

    def panelA_row(p):
        lbl = rc.THETA1_LABELS.get(p, p.replace("_", r"\_"))
        cvals, svals = [lbl], [""]
        for c in cols:
            if c == "logit":
                cc, ss = logit_cell(logit, p)
            else:
                d = stage_data[c]
                names = d.get("param_names_theta1", [])
                if p in names:
                    i = names.index(p)
                    t1se = d.get("theta1_se", []); t1pv = d.get("theta1_pval", [])
                    cc, ss = rc.fmt_coef(d["theta1"][i],
                                         t1se[i] if i < len(t1se) else 0.0,
                                         d.get("G_star"),
                                         t1pv[i] if i < len(t1pv) else None)
                else:
                    cc, ss = "-", ""
            cvals.append(cc); svals.append(ss)
        rows = ["    " + " & ".join(cvals) + r" \\"]
        if any(s.strip() for s in svals[1:]):   # skip an all-blank SE row (e.g. θ₂ SEs not yet computed)
            rows.append("    " + " & ".join(svals) + r" \\")
        rows.append(r"    \addlinespace[0.15ex]")
        return rows

    def panelB_row(lbl):
        is_sigma = lbl.startswith(r"$\Sigma$")           # Σ's are Σ≥0-bounded → boundary handling
        cvals, svals = [lbl], [""]
        for c in cols:
            if c == "logit":
                cvals.append("-"); svals.append(""); continue
            decoded = {l: (v, se, pv) for l, v, se, pv in rc.decode_theta2(stage_data[c])}
            if lbl in decoded:
                v, se, pv = decoded[lbl]
                ob = is_sigma and v is not None and not (isinstance(v, float) and math.isnan(v)) \
                    and abs(v) < rc.SIGMA_BOUND_TOL
                cc, ss = rc.fmt_coef(v, se, stage_data[c].get("G_star"), pv, on_bound=ob)
            else:
                cc, ss = "-", ""
            cvals.append(cc); svals.append(ss)
        rows = ["    " + " & ".join(cvals) + r" \\"]
        if any(s.strip() for s in svals[1:]):   # skip an all-blank SE row (e.g. θ₂ SEs not yet computed)
            rows.append("    " + " & ".join(svals) + r" \\")
        rows.append(r"    \addlinespace[0.15ex]")
        return rows

    L = [
        r"\begin{landscape}",
        r"\begin{spacing}{1.0}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",   # MUST precede \begin{longtable}: a token inside the
        rf"\begin{{longtable}}{{{col_fmt}}}",  # longtable body before \caption starts the first
        rf"    \caption{{BLP Demand: Logit vs.\ RC-BLP stages --- Estimation "  # cell → \caption's \noalign misplaces
        rf"{rc.est_ref(est)}}}",
        rf"    \label{{tab:blp_compare_E{est}_spec12}} \\",
        r"    \toprule",
        f"     & {hdr_cols} \\\\",
        r"    \midrule \endfirsthead",
        rf"    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        f"     & {hdr_cols} \\\\",
        r"    \midrule \endhead",
        r"    \midrule",
        rf"    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols + 1) + r"}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\scriptsize "  # \linewidth = landscape line width (pdflscape clamps \textheight→\textwidth)
        r"\textit{Notes:} The estimation strategy is enumerated in "
        r"Section~\ref{sec:empirical:sleep}. Column 1 is the non-RC logit (`full' sub-model); "
        r"the remaining columns are the RC-BLP stages run on the cluster, each freeing one more random "
        r"coefficient (1 RC $\to$ 5 RC). The logit column reports wild cluster "
        rf"bootstrap standard errors (conglomerate clusters); for the RC columns, {rc_sem}. "
        r"Stars from a Student-$t$ reference with $G^*$ effective clusters: *** $p<0.01$, ** $p<0.05$, "
        r"* $p<0.1$. A $\dagger$ marks a random-coefficient $\Sigma$ estimated at the $\Sigma\ge0$ "
        r"boundary ($\hat\Sigma\approx0$): point reported on the bound, no two-sided SE (Andrews 1999). "
        r"$Q$ is each model's own GMM objective (not comparable across the "
        r"logit/RC boundary --- different moment counts). Mean own-price elasticity is the "
        r"average-market plug-in $\hat\alpha\cdot\overline{\rho(1-s)}$ "
        r"(cf.\ the logit comparison, Table~\ref{tab:demand_logit_spec12_comparison}). "
        r"Spread in percentage points."
        r"} \\",
        r"    \endlastfoot",
        r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel A: Mean utility ($\theta_1$)}} \\",
        r"    \addlinespace[0.3ex]",
    ]
    for p in theta1_params:
        L += panelA_row(p)

    L += [r"    \midrule",
          r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel B: Random coefficients ($\theta_2$)}} \\",
          r"    \addlinespace[0.3ex]"]
    for lbl in rc.build_global_theta2_labels(stage_data):
        L += panelB_row(lbl)

    # ── footer stats ──
    L.append(r"    \midrule")
    def stat_row(name, fn):
        cells = [name]
        for c in cols:
            cells.append(fn(c))
        return "    " + " & ".join(cells) + r" \\"
    def q_of(c):
        d = logit if c == "logit" else stage_data[c]
        q = d.get("Q_value");  return f"${q:.4f}$" if q is not None else "---"
    def n_of(c):
        d = logit if c == "logit" else stage_data[c]
        n = d.get("n_obs");    return f"{n:,}" if n is not None else "---"
    def g_of(c):
        d = logit if c == "logit" else stage_data[c]
        g = d.get("G_star");   return f"{g:.2f}" if g is not None else "---"
    def dim_of(c):
        if c == "logit": return "0"
        return str(len(stage_data[c].get("theta2", [])))
    # Demographics are centered in the engine, so Panel-A θ₁ (incl. spread α) is already the
    # average-market coefficient — directly comparable to the logit α, no effective-α row needed.
    def se_of(c):
        entry = logit if c == "logit" else stage_data[c]
        return rc.semi_elast_cell(entry, est)         # α̂·mean(ρ(1−s)), average-market plug-in
    L += [
        stat_row(r"Mean own-price elasticity", se_of),
        stat_row(r"$Q$ (GMM)", q_of),
        stat_row(r"$\dim(\theta_2)$", dim_of),
        stat_row(r"Observations", n_of),
        stat_row(r"Eff.\ clusters ($G^*$)", g_of),
        r"\end{longtable}",
        r"\end{spacing}",
        r"\end{landscape}",
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Logit-vs-RC-stages landscape table")
    ap.add_argument("--est", type=int)
    ap.add_argument("--routines", default="5,6,7,8")
    args = ap.parse_args()
    ests = [args.est] if args.est else [int(x) for x in args.routines.split(",") if x.strip()]

    for est in ests:
        print(f"\n[E{est}] building Logit-vs-RC comparison table...")
        tex = build_table(est)
        if not tex:
            continue
        fname = f"blp_compare_E{est}_spec12.tex"
        for dest in (TABLES_DIR, DRAFTS_DIR):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / fname).write_text(tex, encoding="utf-8")
            print(f"  saved: {dest / fname}")
    print("\nDone. \\input{blp_compare_E{est}_spec12.tex} (needs \\usepackage{pdflscape,longtable,booktabs,setspace}).")


if __name__ == "__main__":
    main()
