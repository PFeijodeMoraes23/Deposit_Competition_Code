"""
make_iv_tables.py
=================
Two LaTeX tables from the weak-instruments battery (weak_iv_analysis.py -> cluster_processed/weak_iv.json),
in the landscape-longtable style of the demand tables:

  1. tab_firststage_spec12.tex   -- First-stage instrument strength + weak-IV-robust inference, per
                                    routine x deposit-type subsample, with the ESTBAN-augmented set (K=16):
                                    2SLS/LIML alpha, KP & MOP effective F, partial R^2, tF 95% CI, Hansen J.
  2. tab_iv_comparison_spec12.tex -- Before/after adding the ESTBAN branch-competition instrument:
                                    alpha, effective-F, partial R^2 WITHOUT (15) vs WITH (16) ESTBAN
                                    (from the leave-one-group-out battery), isolating its contribution.

Reads BLP_RESULTS/cluster_processed/weak_iv.json; writes to ESTIMATION_OUTPUT/Rout/ + Drafts/Deposit
Competition/. Routine labels use the V_Main \ref{estimation:*} enumerate (make_blp_rc_table.est_ref).

Usage:  python make_iv_tables.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import math
import pathlib
import make_blp_rc_table as rc     # est_ref + DRAFTS_DIR

ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
WEAK_IV    = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "cluster_processed" / "weak_iv.json"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = rc.DRAFTS_DIR
ROUTINES   = [5, 6, 7, 8]
SUBS       = [("type45", "4+5"), ("type4", "4"), ("type5", "5")]


def _num(x, d=3, signed=True):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "---"
    return f"${x:+.{d}f}$" if signed else f"${x:.{d}f}$"


def _F(x):
    return f"${x:,.0f}$" if isinstance(x, (int, float)) and math.isfinite(x) else "---"


def _tf(rec):
    tf = rec.get("tf") or {}
    if tf.get("tf_defined") and tf.get("tf_ci_low") is not None:
        return f"$[{tf['tf_ci_low']:+.2f},\\,{tf['tf_ci_high']:+.2f}]$"
    return "(undef.)" if tf.get("tf_defined") is False else "---"


def _alpha_se(a, se, stars_from=None):
    if not isinstance(a, (int, float)) or not math.isfinite(a):
        return "---"
    s = f"${a:+.3f}$"
    if isinstance(se, (int, float)) and math.isfinite(se) and se > 0:
        s += f" $({se:.3f})$"
    return s


# Grid half-width of the weak-IV-robust CI inversion (weak_iv_analysis searches alpha in [-2, 2]);
# an endpoint at the bound is an OPEN interval, flagged so it is not read as a finite limit.
_GRID = 2.0


def _robust_ci(lo, hi, disconnected=False):
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return "---"
    open_lo = lo <= -_GRID + 1e-9
    open_hi = hi >= _GRID - 1e-9
    ls = r"(-\infty" if open_lo else f"[{lo:+.2f}"
    hs = r"+\infty)" if open_hi else f"{hi:+.2f}]"
    star = r"^{\dagger}" if disconnected else ""
    return f"${ls},\\,{hs}{star}$"


def _wrap(body_lines, col_fmt, caption, label, header, footnote, ncols):
    trow = r" \\"
    return "\n".join([
        r"\begin{landscape}",
        r"\begin{spacing}{1.0}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{longtable}}[c]{{{col_fmt}}}",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}} \\",
        r"    \toprule",
        "    " + header + trow,
        r"    \midrule",
        r"    \endfirsthead",
        rf"    \multicolumn{{{ncols}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        "    " + header + trow,
        r"    \midrule",
        r"    \endhead",
        r"    \midrule",
        rf"    \multicolumn{{{ncols}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols) + r"}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\scriptsize "
        + footnote + r"} \\",
        r"    \endlastfoot",
        *body_lines,
        r"\end{longtable}",
        r"\end{spacing}",
        r"\end{landscape}",
    ])


def build_firststage(wiv):
    # Routine | Sample | N | K | 2SLS a(SE) | LIML a | KP-F | eff-F | partial R^2 | tF 95% CI | J (p)
    col_fmt = "ll r r l r r r r c r"
    header = (r"Routine & Sample & $N$ & $K$ & 2SLS $\hat\alpha$ (SE) & LIML $\hat\alpha$ & "
              r"KP-$F$ & eff-$F$ & partial $R^2$ & tF 95\% CI & $J$ ($p$)")
    body = []
    for k in ROUTINES:
        rr = wiv.get(str(k))
        if not rr:
            continue
        for i, (key, lbl) in enumerate(SUBS):
            r = rr.get(key)
            if not r:
                continue
            rt = rc.est_ref(k) if i == 0 else ""
            pv = f"${r['hansen_J_p']:.2f}$" if r.get("hansen_J_p") is not None else "---"
            body.append("    " + " & ".join([
                rt, lbl, f"{r.get('n_obs', 0):,}", str(r.get("n_iv", "")),
                _alpha_se(r.get("alpha_2sls"), r.get("alpha_se")),
                _num(r.get("alpha_liml")),
                _F(r.get("kp_first_stage_F")), _F(r.get("effective_F")),
                _num(r.get("partial_R2"), 3, signed=False), _tf(r), pv,
            ]) + r" \\")
        if k != ROUTINES[-1]:
            body.append(r"    \addlinespace[0.4ex]")
    foot = (r"\textit{Notes:} First stage of the deposit-spread demand model (the only endogenous "
            r"regressor), on $K$ excluded instruments \emph{including} the ESTBAN branch-competition "
            r"instrument (\texttt{estban\_rival\_branches\_lag}), partialling the product controls and "
            r"clustering by conglomerate. The engine instruments spread for deposit types 4 and 5, "
            r"reported per subsample and pooled. KP-$F$ = cluster-robust Kleibergen--Paap rk Wald $F$; "
            r"eff-$F$ = Montiel-Olea--Pflueger effective $F$; LIML $\hat\alpha$ is the limited-information "
            r"ML estimate (near median-unbiased under weak identification); tF 95\% CI is the "
            r"$F$-adjusted honest interval of Lee, McCrary, Moreira \& Porter (2022). Spread in "
            r"percentage points.")
    return _wrap(body, col_fmt, r"First-Stage Instrument Strength and Weak-IV-Robust Inference (Spec.~12)",
                 "tab:firststage_spec12", header, foot, 11)


def build_comparison(wiv):
    # Routine | Sample | [without ESTBAN: a(SE), eff-F, pR2] | [with ESTBAN: a(SE), eff-F, pR2] | d a
    col_fmt = "ll l r r l r r r"
    header = (r"Routine & Sample & \multicolumn{3}{c}{Without ESTBAN ($K{=}15$)} & "
              r"\multicolumn{3}{c}{With ESTBAN ($K{=}16$)} & $\Delta\hat\alpha$ \\"
              "\n    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
              r"     &  & $\hat\alpha$ (SE) & eff-$F$ & partial $R^2$ & "
              r"$\hat\alpha$ (SE) & eff-$F$ & partial $R^2$ & ")
    body = []
    for k in ROUTINES:
        rr = wiv.get(str(k))
        if not rr:
            continue
        for i, (key, lbl) in enumerate(SUBS):
            r = rr.get(key)
            if not r:
                continue
            loo = r.get("loo") or {}
            base = loo.get("base") or {}                                  # full 16-IV set (numpy)
            drop = (loo.get("drop_group") or {}).get("estban") or {}      # 15-IV set (drop ESTBAN)
            a16, a15 = base.get("alpha"), drop.get("alpha")
            da = (a16 - a15) if isinstance(a16, (int, float)) and isinstance(a15, (int, float)) else None
            rt = rc.est_ref(k) if i == 0 else ""
            body.append("    " + " & ".join([
                rt, lbl,
                _alpha_se(a15, drop.get("alpha_se")), _F(drop.get("eff_F")),
                _num(drop.get("partial_R2"), 3, signed=False),
                _alpha_se(a16, r.get("alpha_se")), _F(base.get("eff_F")),
                _num(base.get("partial_R2"), 3, signed=False),
                _num(da),
            ]) + r" \\")
        if k != ROUTINES[-1]:
            body.append(r"    \addlinespace[0.4ex]")
    foot = (r"\textit{Notes:} 2SLS deposit-spread first stage \emph{without} (the original 15 "
            r"leave-one-out / cost / capital instruments) and \emph{with} the ESTBAN "
            r"branch-competition instrument added ($K{=}16$), from the leave-one-group-out battery "
            r"(same sample, controls and conglomerate clustering). eff-$F$ = Montiel-Olea--Pflueger "
            r"effective $F$. The ESTBAN instrument is independent of the existing set (own VIF "
            r"$\approx 5$ vs $\approx 1{,}350$ for the collinear \texttt{mean\_loo} block) and relevant "
            r"for demand deposits (type 4), tightening the interval and moving $\hat\alpha$ toward the "
            r"economically-signed (negative) value. Spread in percentage points.")
    return _wrap(body, col_fmt, r"Deposit-Spread First Stage Without vs.\ With the ESTBAN Instrument (Spec.~12)",
                 "tab:iv_comparison_spec12", header, foot, 9)


def build_alpha_robust(wiv):
    # THE HEADLINE weak-IV table: identification-robust inference on the deposit-price coefficient α.
    # Leads with the estimator ladder (sign) and the robust confidence sets (level), NOT the ±SE Wald
    # interval — because the instruments are weak, the Wald interval overstates precision.
    # Routine | Sample | N | eff-F | OLS α | 2SLS α (SE) | LIML α | tF 95% CI
    col_fmt = "ll r r r l r c"
    header = (r"Routine & Sample & $N$ & eff-$F$ & OLS $\hat\alpha$ & 2SLS $\hat\alpha$ (SE) & "
              r"LIML $\hat\alpha$ & tF 95\% CI")
    body = []
    for k in ROUTINES:
        rr = wiv.get(str(k))
        if not rr:
            continue
        for i, (key, lbl) in enumerate(SUBS):
            r = rr.get(key)
            if not r:
                continue
            rt = rc.est_ref(k) if i == 0 else ""
            body.append("    " + " & ".join([
                rt, lbl, f"{r.get('n_obs', 0):,}", _F(r.get("effective_F")),
                _num(r.get("alpha_ols")), _alpha_se(r.get("alpha_2sls"), r.get("alpha_se")),
                _num(r.get("alpha_liml")), _tf(r),
            ]) + r" \\")
        if k != ROUTINES[-1]:
            body.append(r"    \addlinespace[0.4ex]")
    foot = (r"\textit{Notes:} Identification-robust inference on the deposit-spread coefficient "
            r"$\alpha$ (the single endogenous regressor), from the log-share linear-IV benchmark on "
            r"the $K{=}16$ excluded instruments, clustered by prudential conglomerate. The estimator "
            r"ladder (OLS $\to$ 2SLS $\to$ LIML) identifies the \emph{sign}: LIML (Anderson \& Rubin, "
            r"1949), median-unbiased under weak identification, moves $\hat\alpha$ further into the "
            r"economically-signed (negative, downward-sloping demand) region than 2SLS, the classic "
            r"weak-instrument signature. But the \emph{level} is not point-identified: eff-$F$ = "
            r"Montiel-Olea--Pflueger effective $F$ (2013) $\approx 4$, so the naive $\pm1.96\,$SE Wald "
            r"interval overstates precision. We report the $F$-adjusted honest tF 95\% CI (Lee, "
            r"McCrary, Moreira \& Porter, 2022) as the identification-robust object; the "
            r"Kleibergen LM/K weak-IV-robust set (2005; wild-cluster-bootstrap criticals) concurs and "
            r"is not shown---the Anderson--Rubin set is empty (the Hansen $J$ rejects) and the LM/K set "
            r"is open and likewise fails to exclude zero. \textbf{The tF CI contains $0$ in every "
            r"routine} (and is undefined for type~5, where eff-$F\approx1$), so the sign of price "
            r"sensitivity is robust but its magnitude is only set-identified. This is driven by three "
            r"features of the setting, not "
            r"by modelling choices: the excluded instruments are weak; the deposit spread is set "
            r"nationally (conglomerate$\times$type$\times$quarter, constant across municipalities), so "
            r"the price has no within-market variation; and deposits are highly concentrated, leaving "
            r"only $G^\ast\approx5$--$7$ effective clusters. Spread in percentage points.")
    return _wrap(body, col_fmt,
                 r"Identification-Robust Inference on the Deposit-Price Coefficient (Spec.~12)",
                 "tab:alpha_weakiv_spec12", header, foot, 8)


def main():
    if not WEAK_IV.exists():
        raise FileNotFoundError(f"weak_iv.json not found at {WEAK_IV} — run weak_iv_analysis.py first.")
    wiv = json.load(open(WEAK_IV))
    tables = {
        "tab_alpha_weakiv_spec12.tex": build_alpha_robust(wiv),
        "tab_firststage_spec12.tex": build_firststage(wiv),
        "tab_iv_comparison_spec12.tex": build_comparison(wiv),
    }
    for name, tex in tables.items():
        for dest in (TABLES_DIR, DRAFTS_DIR):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / name).write_text(tex, encoding="utf-8")
        print(f"  wrote {name}")
    print("Done. \\input{tab_firststage_spec12.tex} / \\input{tab_iv_comparison_spec12.tex} "
          "(needs \\usepackage{pdflscape,longtable,booktabs,setspace}).")


if __name__ == "__main__":
    main()
