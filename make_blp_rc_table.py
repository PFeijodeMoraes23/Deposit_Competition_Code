"""
make_blp_rc_table.py
====================
Generate a LaTeX table from BLP random-coefficient estimation results
(sigma / rc2 / rc3 / rc4 / full / ext1 / ext2 / extended), Spec 12.

Reads blp_results_E{est}_spec_12_{stage}{suffix}.json files produced on the cluster
and dropped into the local BLP_RESULTS folder. The post-coherence-fix cluster runs
write a ``_coherence`` suffix (IFT engine) or ``_coherence_num`` (numerical engine);
this is the DEFAULT here. Use --legacy to read the un-suffixed legacy results.

Usage
-----
  python make_blp_rc_table.py              # E6 coherence (default headline routine)
  python make_blp_rc_table.py --est 3      # E3 coherence
  python make_blp_rc_table.py --all        # E1-E6 coherence
  python make_blp_rc_table.py --engine numerical --est 6   # E6 numerical-engine coherence
  python make_blp_rc_table.py --legacy --est 5             # legacy (un-suffixed) E5

Output
------
  BLP_RESULTS/Rout/blp_rc_E{est}_spec12{suffix}.tex
  Drafts/Deposit Competition/blp_rc_E{est}_spec12{suffix}.tex
"""

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import math
import pathlib
import numpy as np
import scipy.stats as stats

ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RESULTS_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
TABLES_DIR  = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR  = pathlib.Path(
    r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
    r"\Open Finance\Open-Finance\Drafts\Deposit Competition"
)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (must match blp_estimation.jl) ──────────────────────────────────
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
          "log_total_assets_lag"]
COEF_NAMES = ["spread"] + X_COLS   # 1-indexed in Julia

D_COLS = ["gdp_per_capita", "fraction_65plus", "fraction_young",
          "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
          "branches_per1000", "cadunico_families_per1000"]

STAGES = ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]

STAGE_LABELS = {
    "sigma":    r"$\sigma_1$",
    "rc2":      r"RC2",
    "rc3":      r"RC3",
    "rc4":      r"RC4",
    "full":     r"Full",
    "ext1":     r"Ext1",
    "ext2":     r"Ext2",
    "extended": r"Extended",
}

# Human-readable labels for θ₁ parameters
THETA1_LABELS = {
    "alpha":                r"Spread ($\alpha$)",
    "fgc_covered":          r"FGC Covered",
    "has_ip":               r"Has Payment Institution",
    "seg_S2":               r"Segment S2",
    "seg_S3":               r"Segment S3",
    "seg_S4":               r"Segment S4",
    "seg_S5":               r"Segment S5",
    "log_total_assets_lag": r"$\ln(\text{Total Assets}_{t-1})$",
}

# Labels for characteristics in σ/π names
COEF_LABELS = {
    "spread":               r"Spread",
    "fgc_covered":          r"FGC",
    "log_total_assets_lag": r"$\ln$ Assets",
}

DEMO_LABELS = {
    "gdp_per_capita":            r"GDP p.c.",
    "fraction_65plus":           r"Frac.\ Age 65+",
    "connections_per100":        r"Broadband/100",
    "cadunico_families_per1000": r"Cad\'Unico/1000",
}

# ── Label helpers ─────────────────────────────────────────────────────────────

def sigma_label(idx: int) -> str:
    """Julia 1-based index → σ(characteristic) label."""
    name  = COEF_NAMES[idx - 1]
    inner = COEF_LABELS.get(name, name.replace("_", r"\_"))
    return rf"$\sigma$({inner})"


def pi_label(char_idx: int, demo_idx: int) -> str:
    """Julia 1-based indices → π(char × demo) label."""
    char = COEF_NAMES[char_idx - 1]
    demo = D_COLS[demo_idx - 1]
    cl   = COEF_LABELS.get(char, char.replace("_", r"\_"))
    dl   = DEMO_LABELS.get(demo, demo.replace("_", r"\_"))
    return rf"$\pi$({cl} $\times$ {dl})"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_stage(est_id: int, stage: str, suffix: str = "") -> dict | None:
    """Load blp_results_E{est}_spec_12_{stage}{suffix}.json. Returns None if missing/empty.

    ``suffix`` selects the build: "" = legacy, "_coherence" = post-fix IFT engine,
    "_coherence_num" = post-fix numerical engine (matches ENV["BLP_OUTPUT_SUFFIX"] on
    the cluster).
    """
    path = RESULTS_DIR / f"blp_results_E{est_id}_spec_12_{stage}{suffix}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        return data
    except (json.JSONDecodeError, ValueError):
        return None


def decode_theta2(data: dict) -> list[tuple[str, float, float]]:
    """
    Return list of (label, value, se) for each θ₂ parameter in the stage result.
    se is 0.0 when not yet computed.
    """
    sigma_idx  = data.get("sigma_indices", [])
    pi_inter   = data.get("pi_interactions", [])
    theta2     = data.get("theta2", [])
    theta2_se  = data.get("theta2_se", [])

    # theta2_se may be stored as 0.0 when NaN was replaced
    out = []
    for k, sidx in enumerate(sigma_idx):
        val = theta2[k]       if k < len(theta2)    else float("nan")
        se  = theta2_se[k]    if k < len(theta2_se) else 0.0
        out.append((sigma_label(sidx), val, se))

    n_s = len(sigma_idx)
    for j, (ci, di) in enumerate(pi_inter):
        k   = n_s + j
        val = theta2[k]       if k < len(theta2)    else float("nan")
        se  = theta2_se[k]    if k < len(theta2_se) else 0.0
        out.append((pi_label(ci, di), val, se))

    return out


def build_global_theta2_labels(stage_results: dict) -> list[str]:
    """
    Collect all unique θ₂ labels across stages in the order they first appear
    (mirrors the progressive stage construction).
    """
    seen   = []
    for stage in STAGES:
        data = stage_results.get(stage)
        if data is None:
            continue
        for lbl, _, _ in decode_theta2(data):
            if lbl not in seen:
                seen.append(lbl)
    return seen


# ── Formatting ────────────────────────────────────────────────────────────────

def _stars(pval: float) -> str:
    if pval < 0.01:  return r"^{***}"
    if pval < 0.05:  return r"^{**}"
    if pval < 0.10:  return r"^{*}"
    return ""


def fmt_coef(val: float, se: float, G_star: float | None = None) -> tuple[str, str]:
    """Return (coef_cell, se_cell) with significance stars."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "-", ""
    coef_str = f"{val:.4f}"
    if se and se > 0:
        t = val / se
        df = G_star if (G_star and G_star > 1) else None
        pv = 2 * stats.t.sf(abs(t), df=df) if df else 2 * (1 - stats.norm.cdf(abs(t)))
        coef_str += _stars(pv)
        se_str = f"$({se:.4f})$"
    else:
        se_str = ""
    return f"${coef_str}$", se_str


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(est_id: int, suffix: str = "") -> str:
    # Load all available stages
    stage_results = {}
    for s in STAGES:
        d = load_stage(est_id, s, suffix)
        if d is not None:
            stage_results[s] = d

    available = [s for s in STAGES if s in stage_results]
    if not available:
        print(f"[E{est_id}] No result files found in {RESULTS_DIR}")
        return ""

    ncols    = len(available)
    col_fmt  = "l" + "c" * ncols
    hdr_cols = " & ".join(STAGE_LABELS[s] for s in available)

    # Representative data for metadata
    rep = stage_results[available[0]]
    n_obs = rep.get("n_obs") or rep.get("n_clusters", "---")
    G_star_map = {s: stage_results[s].get("G_star") for s in available}

    lines = [
        r"\begin{spacing}{1.0}",
        r"\centering",
        rf"\begin{{longtable}}[c]{{{col_fmt}}}",
        r"    \setlength{\tabcolsep}{5pt}",
        rf"    \caption{{BLP Demand Estimation — E{est_id}, Specification 12}}",
        rf"    \label{{tab:blp_rc_est{est_id}_spec12}} \\",
        r"    \toprule",
        f"    Stage & {hdr_cols} \\\\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        f"    Stage & {hdr_cols} \\\\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols + 1) + r"}{c}{\begin{minipage}{0.85\textwidth}"
        r"\scriptsize \textit{Notes:} Standard errors in parentheses (when available). "
        r"Significance: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
        r"$\theta_1$: mean utility coefficients (linear IV). "
        r"$\theta_2$: random coefficient parameters. "
        r"$Q$: GMM overidentification statistic. "
        r"Spread in percentage points (÷100 from basis points)."
        r"\end{minipage}} \\",
        r"    \endlastfoot",
        "",
    ]

    # ── Panel A: θ₁ (mean utility) ───────────────────────────────────────────
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel A: Mean Utility ($\theta_1$)}} \\")
    lines.append(r"    \addlinespace[0.3ex]")

    # Use param_names from first available stage
    first_data = stage_results[available[0]]
    param_names = first_data.get("param_names_theta1", [])
    # Fallback: try "param_names"
    if not param_names:
        param_names = first_data.get("param_names", [])

    if param_names:
        theta1_display = [p for p in param_names]
    else:
        # Reconstruct from X_COLS
        theta1_display = ["alpha"] + X_COLS

    for p in theta1_display:
        lbl  = THETA1_LABELS.get(p, p.replace("_", r"\_"))
        cvals, svals = [lbl], [""]
        for s in available:
            d      = stage_results[s]
            pnames = d.get("param_names_theta1") or d.get("param_names", [])
            t1     = d.get("theta1", [])
            t1_se  = d.get("theta1_se", [])
            G      = G_star_map[s]
            if p in pnames:
                i  = pnames.index(p)
                v  = t1[i]    if i < len(t1)    else float("nan")
                se = t1_se[i] if i < len(t1_se) else 0.0
                c, s_ = fmt_coef(v, se, G)
                cvals.append(c); svals.append(s_)
            else:
                cvals.append("-"); svals.append("")
        lines.append("    " + " & ".join(cvals) + r" \\")
        lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Panel B: θ₂ (random coefficients) ────────────────────────────────────
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel B: Random Coefficients ($\theta_2$)}} \\")
    lines.append(r"    \addlinespace[0.3ex]")

    global_labels = build_global_theta2_labels(stage_results)

    for lbl in global_labels:
        cvals, svals = [lbl], [""]
        for s in available:
            decoded = {l: (v, se) for l, v, se in decode_theta2(stage_results[s])}
            G = G_star_map[s]
            if lbl in decoded:
                v, se = decoded[lbl]
                c, s_ = fmt_coef(v, se, G)
                cvals.append(c); svals.append(s_)
            else:
                cvals.append("-"); svals.append("")
        lines.append("    " + " & ".join(cvals) + r" \\")
        lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Footer statistics ─────────────────────────────────────────────────────
    lines.append(r"    \midrule")

    q_vals, conv_vals, nobs_vals, gstar_vals = [], [], [], []
    for s in available:
        d = stage_results[s]
        qv   = d.get("Q_value")
        conv = d.get("converged", False)
        nob  = d.get("n_obs")
        G    = d.get("G_star")
        q_vals.append(   f"${qv:.4f}$"     if qv   is not None else "---")
        conv_vals.append("Yes"              if conv             else "No")
        nobs_vals.append(f"{nob:,}"         if nob  is not None else "---")
        gstar_vals.append(f"{G:.2f}"        if G    is not None else "---")

    lines += [
        "    $Q$ (GMM) & "       + " & ".join(q_vals)    + r" \\",
        "    Converged & "        + " & ".join(conv_vals) + r" \\",
        "    Observations & "     + " & ".join(nobs_vals) + r" \\",
        r"    Eff.\ Clusters ($G^*$) & " + " & ".join(gstar_vals) + r" \\",
    ]

    lines += [
        r"\end{longtable}",
        r"\end{spacing}",
    ]

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLP RC LaTeX table generator")
    parser.add_argument("--est", type=int, default=6,
                        help="Estimation strategy (default: 6, the headline single-index routine)")
    parser.add_argument("--all", action="store_true",
                        help="Generate for E1-E6 (coherence) or E1-E5 (--legacy)")
    parser.add_argument("--engine", choices=["ift", "numerical"], default="ift",
                        help="Coherence engine whose results to read (ift→_coherence, "
                             "numerical→_coherence_num). Ignored with --legacy.")
    parser.add_argument("--coherence", action="store_true", default=True,
                        help="Read post-coherence-fix cluster results (default).")
    parser.add_argument("--legacy", dest="coherence", action="store_false",
                        help="Read legacy un-suffixed results instead.")
    args = parser.parse_args()

    # Result-file suffix (matches ENV["BLP_OUTPUT_SUFFIX"] set by blp_2_rc_coherence.jl).
    if args.coherence:
        suffix = "_coherence_num" if args.engine == "numerical" else "_coherence"
    else:
        suffix = ""

    # Coherence build: 8 routines (E1-E8). Legacy build: 5 routines (E1-E5).
    if args.all:
        est_ids = list(range(1, 9)) if args.coherence else list(range(1, 6))
    else:
        est_ids = [args.est]

    for est_id in est_ids:
        print(f"\n[E{est_id}] Building BLP RC table{' [' + suffix.lstrip('_') + ']' if suffix else ''}...")
        tex = build_table(est_id, suffix)
        if not tex:
            continue

        fname = f"blp_rc_E{est_id}_spec12{suffix}.tex"
        for dest in [TABLES_DIR, DRAFTS_DIR]:
            out = dest / fname
            with open(out, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"  Saved: {out}")

    print("\nDone. Insert into LaTeX with:")
    for est_id in est_ids:
        print(f"  \\input{{blp_rc_E{est_id}_spec12{suffix}.tex}}")


if __name__ == "__main__":
    main()
