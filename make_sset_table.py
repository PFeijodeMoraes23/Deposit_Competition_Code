"""
make_sset_table.py
================================================================================
Turn the Stock-Wright S-curves produced by the CUE engine into a confidence-set table.

Input : cluster_raw/blp_results_E{k}_spec_12_{stage}_sset_cue.json  (one per routine; written by the
        BLP_ALPHA_GRID grid mode in blp_engine_gpu.jl — each point is a CONSTRAINED solve, i.e. α is
        pinned at α₀ and θ₂/β are re-optimised, so the resulting set accounts for θ₂ being estimated).
Output: tab_alpha_sset_spec12_cue.tex  → Drafts/Deposit Competition/
        (a NEW file; nothing existing is touched).

Why this is the right object: the AR/LM sets in weak_iv.json hold δ fixed at δ(θ̂₂) and treat θ̂₂ as
known, so they are conditional-on-θ̂₂. Inverting S(α₀) = min_{θ₂,β} N·ḡ'W(θ)ḡ over a grid gives an
identification-robust set for α in the JOINT model (Stock & Wright 2000).

Two df conventions are reported because the choice is genuinely arguable and they can differ a lot:
  * conservative — χ²_L with L = #instruments (the usual Stock-Wright reference)
  * concentrated — χ²_{L − dim(θ₂) − dim(β)}, crediting the concentrated-out parameters
Criticals are asymptotic χ²; a wild-cluster-bootstrap critical for S is NOT implemented (documented in
the table note), which matters here because G* ≈ 5 effective clusters.

Usage
-----
  python make_sset_table.py                       # E3/E4, stage ext1
  python make_sset_table.py --routines 4 --stage ext1
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import make_blp_rc_table as rc      # RAW_DIR / TABLES_DIR / DRAFTS_DIR / est_ref
from utils import routines as _routines


def _intervals(points, crit):
    """Contiguous α₀ runs with S ≤ crit, as (lo, hi) pairs. A disconnected set is reported as such —
    weak-identification sets legitimately can be, and silently reporting only the hull would overstate
    what the data say."""
    runs, cur = [], None
    for p in points:
        if p["S"] <= crit:
            cur = [p["alpha0"], p["alpha0"]] if cur is None else [cur[0], p["alpha0"]]
        elif cur is not None:
            runs.append(tuple(cur)); cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def _fmt(runs, lo_edge, hi_edge):
    if not runs:
        return "empty"
    txt = " $\\cup$ ".join(f"$[{a:+.2f},\\,{b:+.2f}]$" for a, b in runs)
    open_lo = any(abs(a - lo_edge) < 1e-9 for a, _ in runs)
    open_hi = any(abs(b - hi_edge) < 1e-9 for _, b in runs)
    if open_lo or open_hi:
        txt += r"$^{\dagger}$"      # touches a grid edge → not bounded by the data, by the grid
    return txt


def main():
    ap = argparse.ArgumentParser(description="Stock-Wright S-set table from the CUE grid runs")
    ap.add_argument("--routines", default=_routines.csv(_routines.LINK_ESTS))
    ap.add_argument("--stage", default="ext1")
    args = ap.parse_args()
    routines = [int(x) for x in args.routines.split(",") if x.strip()]

    try:
        from scipy.stats import chi2
    except Exception:
        print("scipy required for the chi2 criticals"); return

    rows, found = [], []
    for k in routines:
        p = rc.RAW_DIR / f"blp_results_E{k}_spec_12_{args.stage}_sset_cue.json"
        if not p.is_file():
            print(f"  E{k}: {p.name} not found — run with CUE_SSET=1 and download"); continue
        j = json.load(open(p))
        pts = sorted(j["points"], key=lambda d: d["alpha0"])
        if not pts:
            print(f"  E{k}: empty grid"); continue
        found.append(k)
        lo_e, hi_e = pts[0]["alpha0"], pts[-1]["alpha0"]
        S = [q["S"] for q in pts]
        amin = pts[int(min(range(len(S)), key=S.__getitem__))]["alpha0"]
        df_c, df_k = j.get("df_conservative"), j.get("df_concentrated")
        crit_c = float(chi2.ppf(0.95, df_c)) if df_c and df_c > 0 else float("nan")
        crit_k = float(chi2.ppf(0.95, df_k)) if df_k and df_k > 0 else float("nan")
        set_c = _fmt(_intervals(pts, crit_c), lo_e, hi_e)
        set_k = _fmt(_intervals(pts, crit_k), lo_e, hi_e)
        nconv = sum(1 for q in pts if not q.get("converged", True))
        print(f"  E{k}: {len(pts)} pts | min S={min(S):.4g} at a={amin:+.3f} | "
              f"chi2_95({df_c})={crit_c:.3g} -> {set_c} | chi2_95({df_k})={crit_k:.3g} -> {set_k}"
              + (f" | {nconv} non-converged pts" if nconv else ""))
        rows.append((k, len(pts), amin, min(S), df_c, crit_c, set_c, df_k, crit_k, set_k, nconv))

    if not rows:
        print("nothing to tabulate."); return

    L = [r"\begin{landscape}", r"\begin{spacing}{1.0}", r"\centering\footnotesize",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{longtable}[c]{@{\extracolsep{\fill}}l r r r l r l}",
         r"    \caption{Stock--Wright Identification-Robust Confidence Sets for the Deposit-Price "
         r"Coefficient (Spec.~12, CUE)}",
         r"    \label{tab:alpha_sset_spec12_cue} \\", r"    \toprule",
         r"    Routine & Grid pts & $\arg\min_\alpha S$ & $\min S$ & 95\% set (cons.) & "
         r"$\chi^2_{95}$ & 95\% set (conc.) \\",
         r"    \midrule", r"    \endfirsthead", r"    \toprule",
         r"    Routine & Grid pts & $\arg\min_\alpha S$ & $\min S$ & 95\% set (cons.) & "
         r"$\chi^2_{95}$ & 95\% set (conc.) \\",
         r"    \midrule", r"    \endhead", r"    \bottomrule",
         r"    \multicolumn{7}{@{}p{\dimexpr0.78\linewidth-2\tabcolsep\relax}@{}}{\scriptsize "
         r"\textit{Notes:} $S(\alpha_0)=\min_{\theta_2,\beta} N\,\bar g'\widehat W(\theta)\bar g$ "
         r"evaluated on a grid of pinned $\alpha_0$, each point a fully re-optimised constrained "
         r"solve, so the inverted set accounts for $\theta_2$ being estimated (Stock \& Wright, 2000) "
         r"--- unlike the conditional AR/LM sets, which hold $\delta$ fixed at "
         r"$\delta(\hat\theta_2)$. The weight is continuously updated, "
         r"$\widehat W(\theta)=\widehat\Omega(\theta)^{-1}$ clustered by conglomerate (CUE; the "
         r"nonlinear analogue of LIML). \emph{Cons.} uses $\chi^2_L$ with $L$ the instrument count; "
         r"\emph{conc.} credits the concentrated-out $\theta_2$ and $\beta$. A $\dagger$ marks a set "
         r"touching a grid endpoint --- bounded by the grid, not by the data. Criticals are "
         r"asymptotic $\chi^2$: no wild-cluster-bootstrap critical for $S$ is implemented, which is a "
         r"real caveat at $G^\ast\approx5$ effective clusters. Spread in percentage points.} \\",
         r"    \endlastfoot"]
    for (k, n, amin, smin, dfc, cc, sc, dfk, ck, sk, _) in rows:
        L.append(f"    {rc.est_ref(k)} & {n} & ${amin:+.3f}$ & ${smin:.3g}$ & {sc} & "
                 f"${cc:.3g}$ & {sk} \\\\")
    L += [r"\end{longtable}", r"\end{spacing}", r"\end{landscape}"]
    tex = "\n".join(L)

    for d in (rc.DRAFTS_DIR,):
        if d.is_dir():
            (d / "tab_alpha_sset_spec12_cue.tex").write_text(tex, encoding="utf-8")
            print(f"  saved: {d / 'tab_alpha_sset_spec12_cue.tex'}")
    print(f"\nDone ({len(found)} routines). \\input{{tab_alpha_sset_spec12_cue.tex}}")


if __name__ == "__main__":
    main()
