"""
make_blp_rc_table.py
====================
Generate a LaTeX table from BLP random-coefficient estimation results
(sigma / rc2 / rc3 / rc4 / full / ext1 / ext2 / extended), Spec 12.

Reads blp_results_E{est}_spec_12_{stage}{suffix}.json files produced on the cluster
and dropped into the local BLP_RESULTS folder. The IFT engine writes un-suffixed
results; the numerical engine appends ``_num`` (matches ENV["BLP_OUTPUT_SUFFIX"] set by
blp_rc.jl). Select which with --engine.

Usage
-----
  python make_blp_rc_table.py              # E3 (default headline routine, IFT engine)
  python make_blp_rc_table.py --est 3      # E3
  python make_blp_rc_table.py --all        # every routine in the lineup
  python make_blp_rc_table.py --engine numerical --est 3   # E3 numerical-engine results

Output
------
  BLP_RESULTS/Rout/blp_rc_E{est}_spec12{suffix}.tex
  Drafts/Deposit Competition/blp_rc_E{est}_spec12{suffix}.tex
"""

from utils.venv_guard import ensure_project_venv
from utils import paths as _paths
from utils import routines as _routines
ensure_project_venv(__file__)

import argparse
import json
import math
import pathlib
import numpy as np
import scipy.stats as stats
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
DATA_DIR   = _paths.PROCESSED
RESULTS_DIR = _paths.blp_results_dir()
# Raw per-stage cluster results live in cluster_raw/ after the 2026-06-25 reorg.
RAW_DIR    = RESULTS_DIR / "cluster_raw"
DEMAND_PREP_DIR = _paths.demand_parquet_dir()                     # demand_{k}_*spec_12.parquet
# These tables are built from CLUSTER artifacts under BLP_RESULTS, which are not written per
# sleepiness vintage, so they resolve to the production tree rather than following SLEEP_OUT_ROOT.
TABLES_DIR  = _paths.estimation_output() / "Rout"
DRAFTS_DIR  = _paths.drafts_dir()
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (must match blp_estimation.jl) ──────────────────────────────────
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
          "log_total_assets_lag", "is_state_owned"]
COEF_NAMES = ["spread"] + X_COLS   # 1-indexed in Julia

D_COLS = ["gdp_per_capita", "fraction_65plus", "fraction_young",
          "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
          "branches_per1000", "cadunico_families_per1000"]

# Engine → result-file suffix. Mirrors ENGINE_SUFFIX in blp_rc.jl, which is the authority; keep the
# two in sync. `cue` is the continuously-updated-GMM variant (the LIML analogue) and is a side-by-side
# robustness engine — its files never collide with the ift headline.
ENGINE_SUFFIX = {"ift": "", "numerical": "_num", "cue": "_cue"}

STAGES = ["sigma", "rc2", "rc3", "rc4", "ext1", "ext2", "extended"]
# NB: `full` (the old 5-RC rung) is intentionally ABSENT — the cluster ladder is
# sigma→rc2→rc3→rc4→ext1→ext2→extended (σ(ln assets) was dropped, so `full`≡rc4 and is no longer run).
# Leaving it in the list silently surfaced a STALE pre-is_state_owned `full` result left in cluster_raw.
# The per-routine RC tables (blp_rc_E*) stop at ext1: ext2/extended are the degenerate rungs where the
# fgc/asset-interaction weak identification inflates every SE ~6-10x (α SE ~0.11 → 0.6-1.2) while leaving
# α essentially unchanged, so those two columns carry no usable information. The Full model still lives in
# the _full demand-comparison / robustness tables, which read it directly.
RC_TABLE_STAGES = STAGES[:STAGES.index("ext1") + 1]   # sigma, rc2, rc3, rc4, ext1

# Columns are labelled by the number of freed random coefficients (1→7), so the complexity ladder is
# legible: "1 RC" = one random coefficient (sigma stage) ... "Full" = all seven (extended model).
# `full` is not a stage (see STAGES above), so ext1 = 5 RC and ext2 = 6 RC.
STAGE_LABELS = {
    "sigma":    r"1 RC", "rc2": r"2 RC", "rc3": r"3 RC", "rc4": r"4 RC",
    "ext1":     r"5 RC", "ext2": r"6 RC", "extended": r"Full",
}

# Human-readable labels for θ₁ parameters. The price coefficient uses the SAME wording as the
# non-RC logit table (blp_logit.jl VAR_MAP) and V_Main eq. (1), where α is the coefficient on
# the price/spread ρ — so the logit and RC columns of the compare table share one row label.
THETA1_LABELS = {
    "alpha":                r"Price coefficient ($\alpha$)",
    "fgc_covered":          r"FGC Covered",
    "has_ip":               r"Group Contains IP",
    "seg_S2":               r"Segment S2",
    "seg_S3":               r"Segment S3",
    "seg_S4":               r"Segment S4",
    "seg_S5":               r"Segment S5",
    "log_total_assets_lag": r"$\ln(\text{Total Assets}_{t-1})$",
    "is_state_owned":       r"State-Owned",
}

# Characteristic labels for the σ/π parameter names — cover ALL θ₁ characteristics so σ(·) labels
# never fall back to a raw underscore-escaped column name.
COEF_LABELS = {
    "spread":               r"Spread",
    "fgc_covered":          r"FGC",
    "has_ip":               r"Group Contains IP",
    "seg_S2":               r"Seg.\ S2",
    "seg_S3":               r"Seg.\ S3",
    "seg_S4":               r"Seg.\ S4",
    "seg_S5":               r"Seg.\ S5",
    "log_total_assets_lag": r"$\ln$ Assets",
    "is_state_owned":       r"State-Owned",
}

# Demographic labels — descriptive names consistent with the sleepiness tables and
# tab:demographic_chars (units dropped: BLP demographics enter STANDARDIZED, D̃=(D−D̄)/σ, so
# "per 1k" / "10k R$" would be misleading). Covers all 8 D_COLS → no raw fallbacks.
DEMO_LABELS = {
    "gdp_per_capita":            r"GDP \textit{per capita}",
    "fraction_65plus":           r"Fraction 65+",
    "fraction_young":            r"Fraction Young",
    "pix_users_pf_per1000":      r"PIX Users",
    "connections_per100":        r"Broadband Connections",
    "frac_4g5g":                 r"4G/5G Share",
    "branches_per1000":          r"Bank Branches",
    "cadunico_families_per1000": r"Cad\'Unico Families",
}

# Estimator identity → the \ref{estimation:*} enumerate labels in V_Main (sec:empirical:sleep),
# EXACTLY as the sleepiness comparison tables (est1-4_spec12_stage2_comparison.tex) reference them.
# The demand routine id (E3) is a code artifact; \ref{estimation:single_idx} renders as the paper's
# estimator number, keeping the demand tables consistent with the text. NO ad-hoc names.
# The map comes from config/routines.toml, whose keys must match V_Main's
# `\item\label{estimation:*}` enumerate exactly: a key absent from it falls through to a literal
# "E{id}", which renders as plain text rather than the paper's strategy number -- and an id
# pointing at a REMOVED label renders as "??". Both fail quietly.
#
# Re-exported here under their historical names because six table generators import
# `ESTIMATION_REF` / `est_ref` / `DRAFTS_DIR` from this module.
ESTIMATION_REF = _routines.ESTIMATION_REF
est_ref = _routines.est_ref

# ── Label helpers ─────────────────────────────────────────────────────────────

def sigma_label(idx: int) -> str:
    """Julia 1-based index → σ(characteristic) label."""
    name  = COEF_NAMES[idx - 1]
    inner = COEF_LABELS.get(name, name.replace("_", r"\_"))
    # Upper-case Σ for the random-coefficient std devs, matching the paper's BLP notation
    # (Σ = std-dev matrix, Π = demographic-interaction matrix).
    return rf"$\Sigma$({inner})"


def pi_label(char_idx: int, demo_idx: int) -> str:
    """Julia 1-based indices → Π(char × demo) label."""
    char = COEF_NAMES[char_idx - 1]
    demo = D_COLS[demo_idx - 1]
    cl   = COEF_LABELS.get(char, char.replace("_", r"\_"))
    dl   = DEMO_LABELS.get(demo, demo.replace("_", r"\_"))
    return rf"$\Pi$({cl} $\times$ {dl})"


def equal_col_fmt(ncols: int, label_w: str = "4.8cm") -> str:
    r"""Column spec: a raggedright label column of width ``label_w`` followed by ``ncols``
    equal-width, centered ``p``-columns that together fill \linewidth (= \textwidth in portrait).
    Equal widths keep the inter-column spacing uniform --- this avoids the "bulging column" gap that
    natural-width ``c`` columns produce when one cell (e.g. a large weak-$\theta_2$ SE) is far wider
    than its neighbours. Assumes ``\tabcolsep`` is set before ``\begin{longtable}``; the ``2(ncols+1)``
    factor is the total left+right column padding subtracted so the row fills the line exactly."""
    total = ncols + 1
    data = (r">{\centering\arraybackslash}p{\dimexpr(\linewidth-" + label_w
            + r"-" + str(2 * total) + r"\tabcolsep)/" + str(ncols) + r"\relax}")
    return r">{\raggedright\arraybackslash}p{" + label_w + r"}" + data * ncols


# ── Data loading ──────────────────────────────────────────────────────────────

def load_stage(est_id: int, stage: str, suffix: str = "") -> dict | None:
    """Load blp_results_E{est}_spec_12_{stage}{suffix}.json. Returns None if missing/empty.

    ``suffix`` selects the engine: "" = IFT, "_num" = numerical (matches
    ENV["BLP_OUTPUT_SUFFIX"] on the cluster).
    """
    path = RAW_DIR / f"blp_results_E{est_id}_spec_12_{stage}{suffix}.json"
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


_RHO_CACHE: dict = {}


def mean_rho_one_minus_s(est_id: int):
    """mean(ρ·(1−s)) over the routine's demand sample: ρ = spread_ann/100, s = share_B_cond if is_B
    else share_D, masked to finite ρ,s and 0≤s<1. Times a column's α̂ this gives the mean own-price
    ELASTICITY ∂ln s/∂ln ρ = α̂·ρ·(1−s) — unit-free, since the spread ρ enters in levels (a
    semi-elasticity would be α̂·(1−s), per pp); average-market plug-in, matches blp_logit.jl.
    None if the parquet is missing."""
    if est_id in _RHO_CACHE:
        return _RHO_CACHE[est_id]
    val = None
    try:
        import pandas as pd
        fs = sorted(DEMAND_PREP_DIR.glob(f"demand_{est_id}_*spec_12.parquet"),
                    key=lambda p: p.stat().st_mtime)
        if fs:
            df   = pd.read_parquet(fs[-1], columns=["spread_ann", "share_D", "share_B_cond", "is_B"])
            is_B = df["is_B"].fillna(False).astype(bool).to_numpy()
            s    = np.where(is_B, df["share_B_cond"].to_numpy(float), df["share_D"].to_numpy(float))
            rho  = df["spread_ann"].to_numpy(float) / 100.0
            m    = np.isfinite(rho) & np.isfinite(s) & (s >= 0.0) & (s < 1.0)
            if m.any():
                val = float(np.mean(rho[m] * (1.0 - s[m])))
    except Exception as e:
        print(f"  [elast] E{est_id}: mean(ρ(1−s)) failed — {e}")
    _RHO_CACHE[est_id] = val
    return val


def alpha_of(entry: dict):
    """θ₁ price coefficient α from a result entry (param_names 'alpha' index, else theta1[0])."""
    t1 = entry.get("theta1", [])
    names = entry.get("param_names_theta1") or entry.get("param_names", [])
    if "alpha" in names and names.index("alpha") < len(t1):
        return float(t1[names.index("alpha")])
    return float(t1[0]) if t1 else None


def semi_elast_cell(entry: dict, est_id: int) -> str:
    """α̂·mean(ρ(1−s)) formatted (the average-market own-price elasticity), or '---'."""
    rho = mean_rho_one_minus_s(est_id)
    a   = alpha_of(entry)
    return f"{a * rho:.3f}" if (rho is not None and a is not None) else "---"


def decode_theta2(data: dict) -> list[tuple[str, float, float, float | None]]:
    """
    Return list of (label, value, se, pval) for each θ₂ parameter in the stage result.
    se is 0.0 and pval None when SEs were not computed (BLP_SE_METHOD unset).
    """
    sigma_idx  = data.get("sigma_indices", [])
    pi_inter   = data.get("pi_interactions", [])
    theta2     = data.get("theta2", [])
    theta2_se  = data.get("theta2_se", [])
    theta2_pv  = data.get("theta2_pval", [])

    def _row(label, k):
        val = theta2[k]    if k < len(theta2)    else float("nan")
        se  = theta2_se[k] if k < len(theta2_se) else 0.0
        pv  = theta2_pv[k] if k < len(theta2_pv) else None
        return (label, val, se, pv)

    out = [_row(sigma_label(sidx), k) for k, sidx in enumerate(sigma_idx)]
    n_s = len(sigma_idx)
    out += [_row(pi_label(ci, di), n_s + j) for j, (ci, di) in enumerate(pi_inter)]
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
        for lbl, *_ in decode_theta2(data):
            if lbl not in seen:
                seen.append(lbl)
    return seen


# ── Formatting ────────────────────────────────────────────────────────────────

def _stars(pval: float) -> str:
    if pval < 0.01:  return r"^{***}"
    if pval < 0.05:  return r"^{**}"
    if pval < 0.10:  return r"^{*}"
    return ""


# σ's are bounded σ≥0; a σ pinned at the boundary (σ̂≈0) has no valid two-sided Wald SE — a
# symmetric ±1.96·SE interval would straddle the inadmissible σ<0 region (Andrews 1999/2001).
# We flag such σ with a dagger and report the point on the bound, no two-sided SE. (These are also
# exactly the directions where the WCB SD is degenerate because ∂s/∂σ=0 at σ=0 — false precision.)
SIGMA_BOUND_TOL = 1e-3


def fmt_coef(val: float, se: float, G_star: float | None = None,
             pval: float | None = None, on_bound: bool = False) -> tuple[str, str]:
    """Return (coef_cell, se_cell) with significance stars.

    Significance uses the stored wild-bootstrap `pval` when available, else a Student-t
    reference with df = G* effective clusters (few-cluster correction) — the same priority as
    the logit tables (blp_logit.jl `format_cell_plain`), so stars mean the same thing across
    the demand tables; the Normal is the last fallback. `on_bound=True` marks a σ pinned at
    the σ≥0 boundary: the point is reported with a dagger and NO two-sided SE/stars
    (Andrews 1999/2001)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "-", ""
    coef_str = f"{val:.4f}"
    if on_bound:
        return rf"${coef_str}^{{\dagger}}$", ""
    if se and se > 0 and not (isinstance(se, float) and math.isnan(se)):
        if pval is not None and not (isinstance(pval, float) and math.isnan(pval)):
            pv = pval                                          # stored WCB Wald p-value
        elif G_star and G_star > 1:
            pv = 2 * stats.t.sf(abs(val / se), df=G_star)      # t(G*): few-cluster reference
        else:
            pv = 2 * (1 - stats.norm.cdf(abs(val / se)))
        coef_str += _stars(pv)
        se_str = f"$({se:.4f})$"
    else:
        se_str = ""
    return f"${coef_str}$", se_str


def se_note(data: dict | None) -> str:
    """SE-method sentence from the stage's `se_method` field. WCB (default) / sandwich name the
    method used on the cluster (BLP_SE_METHOD); 'none' means θ₂ SEs were not computed."""
    m = (data or {}).get("se_method", "none")
    if m == "wcb":
        return (r"WCB standard errors (conglomerate clusters) in parentheses, "
                r"for both $\theta_1$ and $\theta_2$")
    if m == "sandwich":
        return (r"Cluster-robust GMM sandwich standard errors (conglomerate clusters) in "
                r"parentheses, for both $\theta_1$ and $\theta_2$")
    return (r"$\theta_1$ standard errors (cluster-robust, conglomerate) in parentheses; interior "
            r"$\theta_2$ standard errors are pending the wild-cluster-bootstrap cluster run "
            r"(\texttt{SE\_METHOD=wcb})")


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(est_id: int, suffix: str = "") -> str:
    # Load all available stages
    stage_results = {}
    for s in RC_TABLE_STAGES:
        d = load_stage(est_id, s, suffix)
        if d is not None:
            stage_results[s] = d

    available = [s for s in RC_TABLE_STAGES if s in stage_results]
    if not available:
        print(f"[E{est_id}] No result files found in {RESULTS_DIR}")
        return ""

    ncols    = len(available)
    col_fmt  = equal_col_fmt(ncols, "4.8cm")   # equal-width columns fill the line (no bulging column)
    hdr_cols = " & ".join(STAGE_LABELS[s] for s in available)

    # Representative data for metadata
    rep = stage_results[available[0]]
    n_obs = rep.get("n_obs") or rep.get("n_clusters", "---")
    G_star_map = {s: stage_results[s].get("G_star") for s in available}
    sem_note   = se_note(stage_results.get("extended") or rep)

    lines = [
        r"\begin{spacing}{1.0}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",   # MUST precede \begin{longtable} (else it starts the
        rf"\begin{{longtable}}[c]{{{col_fmt}}}",  # first cell and \caption's \noalign misplaces)
        rf"    \caption{{BLP Demand Estimation --- Estimation {est_ref(est_id)}}}",
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
        r"    \multicolumn{" + str(ncols + 1) + r"}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{"  # \linewidth = \textwidth (portrait)
        r"\scriptsize \textit{Notes:} The estimation strategy is enumerated in "
        rf"Section~\ref{{sec:empirical:sleep}}. {sem_note}. Significance from a Student-$t$ "
        r"reference with $G^*$ effective clusters (few-cluster correction): "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
        r"$\theta_1$: mean utility coefficients (linear IV); demographics are centered "
        r"($\tilde D=(D-\bar D)/\sigma$), so $\theta_1$ is the average-market coefficient. "
        r"$\theta_2$: random-coefficient parameters ($\Sigma$ = std.\ dev., $\Pi$ = demographic "
        r"interaction); the $\Sigma$'s are bounded $\Sigma\ge0$. "
        r"A $\dagger$ marks a $\Sigma$ estimated at the boundary ($\hat\Sigma\approx0$): we report "
        r"the point on the bound and \emph{no} two-sided standard error, since a symmetric interval "
        r"would straddle $\Sigma<0$ (Andrews 1999) and the bootstrap is degenerate there. "
        r"$Q$: GMM overidentification statistic. Mean own-price elasticity is the average-market "
        r"plug-in $\hat\alpha\cdot\overline{\rho(1-s)}$. "
        r"Spread in percentage points (÷100 from basis points)."
        r"} \\",
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
            t1_pv  = d.get("theta1_pval", [])
            G      = G_star_map[s]
            if p in pnames:
                i  = pnames.index(p)
                v  = t1[i]    if i < len(t1)    else float("nan")
                se = t1_se[i] if i < len(t1_se) else 0.0
                pv = t1_pv[i] if i < len(t1_pv) else None
                c, s_ = fmt_coef(v, se, G, pv)
                cvals.append(c); svals.append(s_)
            else:
                cvals.append("-"); svals.append("")
        lines.append("    " + " & ".join(cvals) + r" \\")
        if any(s.strip() for s in svals[1:]):   # skip an all-blank SE row (e.g. θ₂ SEs not yet computed)
            lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Panel B: θ₂ (random coefficients) ────────────────────────────────────
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel B: Random Coefficients ($\theta_2$)}} \\")
    lines.append(r"    \addlinespace[0.3ex]")

    global_labels = build_global_theta2_labels(stage_results)

    for lbl in global_labels:
        is_sigma = lbl.startswith(r"$\Sigma$")           # Σ's are Σ≥0-bounded → boundary handling
        cvals, svals = [lbl], [""]
        for s in available:
            decoded = {l: (v, se, pv) for l, v, se, pv in decode_theta2(stage_results[s])}
            G = G_star_map[s]
            if lbl in decoded:
                v, se, pv = decoded[lbl]
                ob = is_sigma and v is not None and not (isinstance(v, float) and math.isnan(v)) \
                    and abs(v) < SIGMA_BOUND_TOL
                c, s_ = fmt_coef(v, se, G, pv, on_bound=ob)
                cvals.append(c); svals.append(s_)
            else:
                cvals.append("-"); svals.append("")
        lines.append("    " + " & ".join(cvals) + r" \\")
        if any(s.strip() for s in svals[1:]):   # skip an all-blank SE row (e.g. θ₂ SEs not yet computed)
            lines.append("    " + " & ".join(svals) + r" \\")
        lines.append(r"    \addlinespace[0.15ex]")

    # ── Footer statistics ─────────────────────────────────────────────────────
    lines.append(r"    \midrule")

    q_vals, nobs_vals, gstar_vals, se_vals = [], [], [], []
    rho = mean_rho_one_minus_s(est_id)
    for s in available:
        d = stage_results[s]
        qv   = d.get("Q_value")
        nob  = d.get("n_obs")
        G    = d.get("G_star")
        q_vals.append(   f"${qv:.4f}$"     if qv   is not None else "---")
        nobs_vals.append(f"{nob:,}"         if nob  is not None else "---")
        gstar_vals.append(f"{G:.2f}"        if G    is not None else "---")
        a = alpha_of(d)
        se_vals.append(f"{a * rho:.3f}" if (rho is not None and a is not None) else "---")

    lines += [
        "    $Q$ (GMM) & "       + " & ".join(q_vals)    + r" \\",
        r"    Mean own-price elasticity & " + " & ".join(se_vals) + r" \\",
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
    parser.add_argument("--est", type=int, default=_routines.LINK_ESTS[0],
                        help=f"Estimation strategy (default: {_routines.LINK_ESTS[0]}, the "
                             "headline single-index routine)")
    parser.add_argument("--all", action="store_true",
                        help="Generate for every routine in the lineup")
    parser.add_argument("--ests", nargs="+", type=int, default=None,
                        help="Generate for an explicit subset, e.g. --ests 3 4. Use this "
                             "rather than --all when some routines' RC chains have not "
                             "reached ext1: a routine that stops earlier still yields a "
                             "table, but a narrower one (fewer stage columns), which is easy "
                             "to mistake for a complete result.")
    parser.add_argument("--engine", choices=["ift", "numerical", "cue"], default="ift",
                        help="Engine whose results to read (ift→un-suffixed, "
                             "numerical→_num, cue→_cue).")
    args = parser.parse_args()

    # Result-file suffix — must match ENGINE_SUFFIX in blp_rc.jl (the single authority).
    # It flows into the OUTPUT filename too, so a non-ift engine writes a NEW .tex beside the
    # headline rather than overwriting it.
    suffix = ENGINE_SUFFIX[args.engine]

    if args.ests:
        est_ids = args.ests
    elif args.all:
        est_ids = list(_routines.ACTIVE)
    else:
        est_ids = [args.est]

    for est_id in est_ids:
        print(f"\n[E{est_id}] Building BLP RC table{' [' + suffix.lstrip('_') + ']' if suffix else ''}...")
        tex = build_table(est_id, suffix)
        if not tex:
            continue

        fname = f"blp_rc_E{est_id}_spec12{suffix}.tex"
        for dest in [DRAFTS_DIR]:
            out = dest / fname
            with open(out, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"  Saved: {out}")

    print("\nDone. Insert into LaTeX with:")
    for est_id in est_ids:
        print(f"  \\input{{blp_rc_E{est_id}_spec12{suffix}.tex}}")


if __name__ == "__main__":
    main()
