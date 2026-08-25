"""
make_iv_tables.py
=================
Two LaTeX tables from the weak-instruments battery (weak_iv_analysis.py -> cluster_processed/weak_iv.json),
in the landscape-longtable style of the demand tables:

  1. tab_firststage_spec12.tex   -- First-stage instrument strength, one line per routine on the
                                    engine's estimation sample, with the ESTBAN-augmented set (K=16):
                                    2SLS alpha, KP F and MOP effective F.
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
from utils import paths as _paths
from utils import routines as _routines
import make_blp_rc_table as rc     # est_ref + DRAFTS_DIR
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
WEAK_IV    = _paths.blp_results_dir() / "cluster_processed" / "weak_iv.json"
WEAK_IV_EXT1 = WEAK_IV.parent / "weak_iv_ext1.json"          # structural-δ battery (--delta-stage ext1)
DIAG_MR      = WEAK_IV.parent / "diag_moment_reduction.json"  # step_moment_reduction.py output
TABLES_DIR = _paths.estimation_output() / "Rout"   # cluster-derived, not per-vintage
DRAFTS_DIR = rc.DRAFTS_DIR
ROUTINES   = list(_routines.ACTIVE)
SUBS       = [("all", "all"), ("type12", "1+2"), ("type45", "4+5"),
              ("type4", "4"), ("type5", "5")]
# Tables 1 and 2 report one line per routine, on the engine's own estimation sample. The per-block
# split stays in the ESTBAN comparison table (below), where the block is what is being compared.
SUBS_ALL   = [("all", "all")]


def _num(x, d=3, signed=True):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "---"
    return f"${x:+.{d}f}$" if signed else f"${x:.{d}f}$"


def _F(x):
    return f"${x:,.0f}$" if isinstance(x, (int, float)) and math.isfinite(x) else "---"


def _F1(x):
    # Tables 1 and 2 report one row per routine, all with eff-F near 4; rounding to the nearest
    # integer collapses 4.20/3.95/4.23/4.11 onto a single "4" and hides the spread across routines.
    return f"${x:,.1f}$" if isinstance(x, (int, float)) and math.isfinite(x) else "---"


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


def _wrap(body_lines, col_fmt, caption, label, header, footnote, ncols, width=r"0.70\linewidth",
          landscape=True):
    # `width` is the width of the notes \multicolumn, and therefore of the whole table: longtable sizes
    # itself from its widest row, and the notes row is always the widest. At the full landscape
    # \linewidth (~24.7cm) the columns hold ~16cm of content, and longtable parks the entire ~9cm of
    # slack in the LAST inter-column gap -- a chasm before the final column (@{\extracolsep{\fill}} does
    # not redistribute it here). Sizing the notes near the natural content width removes the slack at
    # the source; \extracolsep spreads whatever little remains.
    trow = r" \\"
    return "\n".join([
        *([r"\begin{landscape}"] if landscape else []),
        r"\begin{spacing}{1.0}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        # @{\extracolsep{\fill}}: the notes row below is a \multicolumn spanning \linewidth, which
        # stretches the table to the full line. Without a stretch directive longtable dumps ALL of that
        # slack into the last inter-column gap (leaving a chasm before the final column); \extracolsep
        # {\fill} spreads it evenly across every gap instead.
        rf"\begin{{longtable}}[c]{{@{{\extracolsep{{\fill}}}}{col_fmt}}}",
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
        r"    \multicolumn{" + str(ncols) + r"}{@{}p{\dimexpr" + width + r"-2\tabcolsep\relax}@{}}{\scriptsize "
        + footnote + r"} \\",
        r"    \endlastfoot",
        *body_lines,
        r"\end{longtable}",
        r"\end{spacing}",
        *([r"\end{landscape}"] if landscape else []),
    ])


def build_firststage(wiv):
    # Routine | N | K | 2SLS a(SE) | KP-F | eff-F
    col_fmt = "l r r l r r"
    header = (r"Routine & $N$ & $K$ & 2SLS $\hat\alpha$ (SE) & KP-$F$ & eff-$F$")
    body = []
    for k in ROUTINES:
        rr = wiv.get(str(k))
        if not rr:
            continue
        for i, (key, lbl) in enumerate(SUBS_ALL):
            r = rr.get(key)
            if not r:
                continue
            rt = rc.est_ref(k) if i == 0 else ""
            body.append("    " + " & ".join([
                rt, f"{r.get('n_obs', 0):,}", str(r.get("n_iv", "")),
                _alpha_se(r.get("alpha_2sls"), r.get("alpha_se")),
                _F1(r.get("kp_first_stage_F")), _F1(r.get("effective_F")),
            ]) + r" \\")
    foot = (r"\textit{Notes:} First stage of the deposit-spread demand model (the only endogenous "
            r"regressor), on $K$ excluded instruments \emph{including} the ESTBAN branch-competition "
            r"instrument (\texttt{estban\_rival\_branches\_lag}), partialling the product controls and "
            r"clustering by conglomerate. Each row is the engine's own estimation sample, on which "
            r"$\alpha$ is estimated. \emph{Within deposit types~1--2---the regulated products, $61\%$ of "
            r"rows---the engine does not instrument the spread}: those rates are regulated or fixed at "
            r"zero and enter raw. The 2SLS entry here therefore describes what a uniform IV estimator "
            r"would do on the full sample, not what the engine does; it is reported at that scope "
            r"because a diagnostic confined to the instrumented types~4--5 would cover only $39\%$ of "
            r"the sample. KP-$F$ = cluster-robust Kleibergen--Paap rk Wald $F$; eff-$F$ = "
            r"Montiel-Olea--Pflueger effective $F$, against a critical value of $\approx9.4$ for "
            r"tolerating even a $30\%$ worst-case bias. Spread in percentage points.")
    return _wrap(body, col_fmt, r"First-Stage Instrument Strength and Weak-IV-Robust Inference (Spec.~12)",
                 "tab:firststage_spec12", header, foot, 6)


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
    # Routine | N | eff-F | 2SLS α (SE) | LIML α | tF 95% CI
    col_fmt = "l r r l r c"
    header = (r"Routine & $N$ & eff-$F$ & 2SLS $\hat\alpha$ (SE) & "
              r"LIML $\hat\alpha$ & tF 95\% CI")
    body = []
    for k in ROUTINES:
        rr = wiv.get(str(k))
        if not rr:
            continue
        for i, (key, lbl) in enumerate(SUBS_ALL):
            r = rr.get(key)
            if not r:
                continue
            rt = rc.est_ref(k) if i == 0 else ""
            body.append("    " + " & ".join([
                rt, f"{r.get('n_obs', 0):,}", _F1(r.get("effective_F")),
                _alpha_se(r.get("alpha_2sls"), r.get("alpha_se")),
                _num(r.get("alpha_liml")), _tf(r),
            ]) + r" \\")
    foot = (r"\textit{Notes:} Identification-robust inference on the deposit-spread coefficient "
            r"$\alpha$ (the single endogenous regressor), from the log-share linear-IV benchmark on "
            r"the $K{=}16$ excluded instruments, clustered by prudential conglomerate, on the engine's "
            r"own estimation sample. \emph{This design does not identify} $\alpha$. The "
            r"Montiel-Olea--Pflueger effective $F$ (2013) is $\approx4.1$ here, against a critical "
            r"value of $\approx9.4$ for tolerating even a $30\%$ worst-case bias; restricted to the "
            r"two products the engine actually instruments (types~4--5, $39\%$ of rows) it falls to "
            r"$\approx1.4$. At the sample strength the $F$-adjusted honest tF $95\%$ interval of Lee, "
            r"McCrary, Moreira \& Porter (2022) \emph{is} defined --- bounded, wide, and containing "
            r"zero --- whereas on types~4--5 alone it is undefined, the first stage falling below the "
            r"range that adjustment is tabulated over. LIML $\hat\alpha$ is the limited-information ML "
            r"estimate, near median-unbiased under weak identification; it diverges here, which is "
            r"itself the diagnostic. The Anderson--Rubin set (1949) is \emph{empty}, mechanically: "
            r"$\min_\alpha$AR is the Hansen $J$, which rejects the overidentifying restrictions "
            r"($p\le0.002$; a low bar at $n\approx200{,}000$ with 15 restrictions). The Kleibergen "
            r"(2005) LM/K set, which does not collapse under that rejection, contains zero. "
            r"Trimming the instrument set raises eff-$F$ to $\approx5.0$ (the five leave-one-out rival "
            r"characteristics alone), confirming many-weak-instrument dilution, but leaves it below "
            r"threshold with partial $R^2\approx0.001$. Three features of the setting drive this, not "
            r"modelling choices: the excluded instruments are weak and near-collinear (Patnaik effective "
            r"$K\approx3.5$ of 16); the deposit spread is set nationally "
            r"(conglomerate$\times$type$\times$quarter, constant across municipalities), so the price "
            r"has no within-market variation; and deposits are highly concentrated, leaving only "
            r"$G^\ast\approx5$ effective clusters. Note that this is \emph{not} the structural "
            r"$\hat\alpha$: it inverts the log-share (plain logit) $\delta$ rather than "
            r"$\delta(\hat\theta_2)$, and it instruments the spread on every row, whereas the engine "
            r"leaves the regulated types~1--2 raw. The random-coefficients estimates, which identify "
            r"$\alpha$ off the share inversion rather than these instruments alone, are reported "
            r"separately. Spread in percentage points.")
    return _wrap(body, col_fmt,
                 r"Identification-Robust Inference on the Deposit-Price Coefficient (Spec.~12)",
                 "tab:alpha_weakiv_spec12", header, foot, 6)


def _diag_variant(diag, k, name):
    """Look up a variant dict in the diag_moment_reduction.json list; {} when absent/stale."""
    for r in diag:
        if r.get("routine") == k:
            for v in r.get("variants") or []:
                if v.get("variant") == name:
                    return v
    return {}


def _ji_set(lo, hi, disconnected=False, empty=False):
    if empty:
        return r"$\emptyset$"
    return _robust_ci(lo, hi, disconnected)


def build_ji(wiv, wiv_ext1, diag):
    # The just-identified repair: one moment (the fitted first stage), so no weight matrix, no
    # Hansen J, and the AR inversion cannot be emptied by overidentification failure. Three δ/
    # instrument conventions side by side, all with the SAME WCB machinery (Webb, B=999, seed 0,
    # null imposed, Ω̂ observed per grid point). The engine-grade column is the identification-
    # robust CI for the estimator the paper actually reports (FWL: the engine's θ₁ step IS
    # just-identified IV with instrument spread_hat).
    col_fmt = "l l c c c"
    header = (r"Routine & Engine $\hat\alpha$ (SE) & Benchmark JI--WCB & Structural JI--WCB & "
              r"Engine-grade JI--WCB")
    body = []
    for k in ROUTINES:
        rb = (wiv.get(str(k)) or {}).get("all") or {}
        rs = (wiv_ext1.get(str(k)) or {}).get("all") or {}
        ve = _diag_variant(diag, k, "ji_engine")
        cells = [rc.est_ref(k),
                 _alpha_se(ve.get("alpha_2sls"), ve.get("alpha_se")),
                 (_ji_set(rb.get("ji_ci_low"), rb.get("ji_ci_high"),
                          rb.get("ji_ci_disconnected", False)) if rb else "---"),
                 (_ji_set(rs.get("ji_ci_low"), rs.get("ji_ci_high"),
                          rs.get("ji_ci_disconnected", False)) if rs else "---")]
        sw = ve.get("sset_wcb") or {}
        cells.append(_ji_set(sw.get("lo"), sw.get("hi"), sw.get("disconnected", False),
                             sw.get("empty", False)) if sw else "---")
        body.append("    " + " & ".join(cells) + r" \\")
    foot = (r"\textit{Notes:} Identification-robust $95\%$ confidence sets for the deposit-spread "
            r"coefficient $\alpha$ under the \emph{just-identified} design: the sixteen excluded "
            r"instruments are collapsed to the single fitted first stage $z^{*}$, so there is no "
            r"weight matrix, no Hansen $J$, and the Anderson--Rubin inversion cannot be emptied by "
            r"an overidentification rejection --- the set contains the point estimate by "
            r"construction. All three set columns use the same inference: grid inversion over "
            r"$\alpha\in[-2,+2]$ against wild-cluster-bootstrap criticals (Webb six-point weights, "
            r"$B{=}999$, null imposed, the moment variance held at its observed value at each grid "
            r"point), clustered by prudential conglomerate --- the few-cluster-valid treatment "
            r"($G^\ast\approx5.5$ effective clusters). Columns differ only in the regressand and "
            r"instrument convention: \emph{Benchmark} inverts the log-share (plain logit) $\delta$ "
            r"with a uniform first stage on all rows; \emph{Structural} replaces the regressand "
            r"with the Berry-inverted $\delta(\hat\theta_2)$ of the reported (ext1) fit, same "
            r"uniform first stage; \emph{Engine-grade} uses $\delta(\hat\theta_2)$ with the "
            r"engine's own generated instrument --- the per-type projected spread for deposit "
            r"types~4--5 and the \emph{raw} spread for the regulated types~1--2, no intercept. By "
            r"Frisch--Waugh--Lovell the engine's $\theta_1$ step is exactly just-identified IV with "
            r"that instrument, so the Engine $\hat\alpha$ column reproduces the reported structural "
            r"price coefficient (verified to $5\times10^{-6}$) and the engine-grade set is the "
            r"robust interval for \emph{the estimator the paper reports}; it also makes explicit "
            r"that the regulated products' spread is treated as exogenous. Caveats: $z^{*}$ is "
            r"fitted in sample, and a cluster-level two-fold cross-fit variant "
            r"(\texttt{ji\_split}, diagnostics file) shows the uniform instrument's strength is "
            r"substantially optimistic --- the sets here are conditional on $\hat\theta_2$ and on "
            r"the generated instrument. A set printed as open at $\pm2$ is bounded by the search "
            r"window, not the data; $\dagger$ marks a disconnected set. Spread in percentage "
            r"points.")
    return _wrap(body, col_fmt,
                 r"Just-Identified Robust Confidence Sets for the Deposit-Price Coefficient (Spec.~12)",
                 "tab:alpha_ji_spec12", header, foot, 5, width=r"0.92\linewidth", landscape=False)


def main():
    if not WEAK_IV.exists():
        raise FileNotFoundError(f"weak_iv.json not found at {WEAK_IV} — run weak_iv_analysis.py first.")
    wiv = json.load(open(WEAK_IV))
    # The JI table's two extra inputs are allowed to be missing or stale (the generator must not go
    # red because a diagnostic upstream has not been re-run) — absent entries render as ---.
    wiv_ext1, diag = {}, []
    if WEAK_IV_EXT1.exists():
        try:
            wiv_ext1 = json.load(open(WEAK_IV_EXT1))
        except Exception as e:
            print(f"  WARNING: {WEAK_IV_EXT1.name} unreadable ({e}) — structural column will be ---")
    else:
        print(f"  WARNING: {WEAK_IV_EXT1.name} missing — structural column will be ---")
    if DIAG_MR.exists():
        try:
            diag = json.load(open(DIAG_MR))
        except Exception as e:
            print(f"  WARNING: {DIAG_MR.name} unreadable ({e}) — engine-grade column will be ---")
    else:
        print(f"  WARNING: {DIAG_MR.name} missing — engine-grade column will be ---")
    tables = {
        "tab_alpha_weakiv_spec12.tex": build_alpha_robust(wiv),
        "tab_firststage_spec12.tex": build_firststage(wiv),
        "tab_iv_comparison_spec12.tex": build_comparison(wiv),
        "tab_alpha_ji_spec12.tex": build_ji(wiv, wiv_ext1, diag),
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
