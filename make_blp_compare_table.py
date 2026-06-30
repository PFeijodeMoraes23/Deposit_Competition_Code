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
  python make_blp_compare_table.py            # E5-E8 (the cluster default set)
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

ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RES_DIR    = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
RAW_DIR    = RES_DIR / "cluster_raw"
LOGIT_DIR  = RES_DIR / "logit"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = rc.DRAFTS_DIR
TABLES_DIR.mkdir(parents=True, exist_ok=True)

STAGES = rc.STAGES
# Column headers for the RC stages (clearer than the per-stage σ labels).
STAGE_HEAD = {"sigma": "Sigma", "rc2": "RC2", "rc3": "RC3", "rc4": "RC4",
              "full": "Full", "ext1": "Ext1", "ext2": "Ext2", "extended": "Extended"}


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
def _load_eff_alpha():
    p = RES_DIR / "cluster_processed" / "effective_alpha.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
EFF_ALPHA = _load_eff_alpha()   # {routine: {alpha_bar, alpha_i_mean, ...}} from process_blp_outputs.py


def build_table(est: int) -> str:
    stage_data = {s: load_stage(est, s) for s in STAGES}
    stage_data = {s: d for s, d in stage_data.items() if d is not None}
    avail = [s for s in STAGES if s in stage_data]
    if not avail:
        print(f"[E{est}] no RC stage results in {RAW_DIR}")
        return ""
    logit = load_logit_full(est)

    cols = (["logit"] if logit else []) + avail          # column keys, left→right
    ncols = len(cols)
    head = []
    for c in cols:
        head.append("Logit" if c == "logit" else STAGE_HEAD[c])
    hdr_cols = " & ".join(head)
    col_fmt = "l" + "c" * ncols

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
                    cc, ss = rc.fmt_coef(d["theta1"][i], d.get("theta1_se", [])[i]
                                         if i < len(d.get("theta1_se", [])) else 0.0,
                                         d.get("G_star"))
                else:
                    cc, ss = "-", ""
            cvals.append(cc); svals.append(ss)
        return ["    " + " & ".join(cvals) + r" \\",
                "    " + " & ".join(svals) + r" \\",
                r"    \addlinespace[0.15ex]"]

    def panelB_row(lbl):
        cvals, svals = [lbl], [""]
        for c in cols:
            if c == "logit":
                cvals.append("-"); svals.append(""); continue
            decoded = {l: (v, se) for l, v, se in rc.decode_theta2(stage_data[c])}
            if lbl in decoded:
                v, se = decoded[lbl]
                cc, ss = rc.fmt_coef(v, se, stage_data[c].get("G_star"))
            else:
                cc, ss = "-", ""
            cvals.append(cc); svals.append(ss)
        return ["    " + " & ".join(cvals) + r" \\",
                "    " + " & ".join(svals) + r" \\",
                r"    \addlinespace[0.15ex]"]

    L = [
        r"\begin{landscape}",
        r"\begin{spacing}{1.0}",
        r"\centering\scriptsize",
        rf"\begin{{longtable}}{{{col_fmt}}}",
        r"    \setlength{\tabcolsep}{4pt}",
        rf"    \caption{{BLP Demand: Logit vs.\ RC-BLP stages --- E{est} "
        rf"({rc_label(est)}), Specification 12}}",
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
        r"    \multicolumn{" + str(ncols + 1) + r"}{p{0.95\textwidth}}{\scriptsize "
        r"\textit{Notes:} Column 1 is the non-RC logit (`full' sub-model); the remaining "
        r"columns are the RC-BLP stages run on the cluster, each freeing one more random "
        r"coefficient (Sigma$=$1 $\sigma$ $\to$ Extended$=$8). SEs in parentheses; "
        r"$\theta_1$ SEs are cluster-robust (logit) / analytic (RC), $\theta_2$ SEs are not "
        r"computed (RC point estimates, no stars). Stars: *** $p<0.01$, ** $p<0.05$, "
        r"* $p<0.1$. $Q$ is each model's own GMM objective (not comparable across the "
        r"logit/RC boundary --- different moment counts). Spread in percentage points."
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
    def conv_of(c):
        d = logit if c == "logit" else stage_data[c]
        return "Yes" if d.get("converged") else "No"
    def n_of(c):
        d = logit if c == "logit" else stage_data[c]
        n = d.get("n_obs");    return f"{n:,}" if n is not None else "---"
    def g_of(c):
        d = logit if c == "logit" else stage_data[c]
        g = d.get("G_star");   return f"{g:.2f}" if g is not None else "---"
    def dim_of(c):
        if c == "logit": return "0"
        return str(len(stage_data[c].get("theta2", [])))
    eff_rec = EFF_ALPHA.get(str(est))
    def eff_of(c):
        # Logit alpha is already an average-market coefficient; RC alpha_bar (Panel A) is at
        # demographics=0, so the comparable quantity is the mean EFFECTIVE alpha_i over markets.
        if c == "logit":
            t1 = (logit or {}).get("theta1") or []
            return f"${t1[0]:+.3f}$" if t1 else "---"
        if c == "extended" and eff_rec:
            return f"${eff_rec['alpha_i_mean']:+.3f}$"
        return "---"
    L += [
        stat_row(r"$Q$ (GMM)", q_of),
        stat_row(r"$\dim(\theta_2)$", dim_of),
        stat_row(r"Eff.\ $\bar\alpha$ (mean, real mkts)", eff_of),
        stat_row(r"Converged", conv_of),
        stat_row(r"Observations", n_of),
        stat_row(r"Eff.\ clusters ($G^*$)", g_of),
        r"\end{longtable}",
        r"\end{spacing}",
        r"\end{landscape}",
    ]
    return "\n".join(L)


def rc_label(est: int) -> str:
    return {5: "Pooled Single-Index", 6: "Single-Index + Time",
            7: "Joint Single-Index", 8: "Joint + Time"}.get(est, f"E{est}")


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
