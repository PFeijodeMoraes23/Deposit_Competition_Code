"""
make_iv_sleep_tables.py
=======================
LaTeX tables from the SLEEPINESS (phi) first-stage weak-instruments battery
(weak_iv_sleep_analysis.py -> cluster_processed/weak_iv_sleep.json), in the landscape-longtable
style of make_iv_tables.py:

  1. tab_firststage_sleep.tex  -- First-stage instrument strength per instrument spec x deposit-type
                                  subsample, BOTH pooled and two-way-FE-demeaned: N, K, joint F
                                  (pooled / two-way), MOP effective-F (two-way), partial R^2 (two-way),
                                  Hansen J(p). The pooled-strong / within-moderate contrast is the story.
  2. tab_alpha_weakiv_sleep.tex -- Identification-robust inference: the LINEAR-IV PROJECTION benchmark
                                  (2SLS of two-way-FE deposit_balance on spread_qoq) OLS/2SLS/LIML alpha
                                  + tF 95% CI, alongside the STRUCTURAL control-function coefficient
                                  gamma (WCB t). Footnote states in bold that alpha is a linear projection
                                  benchmark, NOT phi, and flags the G* ~ 5-7 precision caveat.
  3. tab_phi_need_sleep.tex     -- The phi "do we need it" materiality test (spec 12): logit vs sieve
                                  link, national deposit-weighted phi_t OLS vs IV_HausmanFull.

Reads BLP_RESULTS/cluster_processed/weak_iv_sleep.json; writes to ESTIMATION_OUTPUT/Rout/ + Drafts/
Deposit Competition/. Reuses make_iv_tables' formatting helpers (_wrap/_num/_F/_tf/_alpha_se).

Usage:  python make_iv_sleep_tables.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import math
import pathlib

import make_blp_rc_table as rc                    # DRAFTS_DIR
from make_iv_tables import _wrap, _num, _F, _tf, _alpha_se   # reuse the formatting scaffold

ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
WEAK_IV    = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "cluster_processed" / "weak_iv_sleep.json"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = rc.DRAFTS_DIR

SPEC_ORDER = ["IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]
SPEC_LABEL = {"IV_CostShifters": r"Cost shifters (3)",
              "IV_Wholesale":    r"+ Wholesale (6)",
              "IV_HausmanFull":  r"+ Hausman LOO (7)"}
SUBS = [("type45", "4+5"), ("type4", "4"), ("type5", "5")]


def _Ff(x, d=1):
    return f"${x:,.{d}f}$" if isinstance(x, (int, float)) and math.isfinite(x) else "---"


def _robust_ci_grid(lo, hi, grid_lo, grid_hi, disconnected=False):
    """Weak-IV-robust CI cell, detecting OPEN endpoints against the PER-ROW alpha grid (the sleep grid
    is rescaled per subsample, so the demand hard-coded +-2 does not apply)."""
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return "---"
    gl = grid_lo if isinstance(grid_lo, (int, float)) else -2.0
    gh = grid_hi if isinstance(grid_hi, (int, float)) else 2.0
    open_lo = lo <= gl + 1e-9
    open_hi = hi >= gh - 1e-9
    ls = r"(-\infty" if open_lo else f"[{lo:+.2f}"
    hs = r"+\infty)" if open_hi else f"{hi:+.2f}]"
    star = r"^{\dagger}" if disconnected else ""
    return f"${ls},\\,{hs}{star}$"


def _gamma_t(cf):
    if not cf or not isinstance(cf.get("gamma"), (int, float)) or not math.isfinite(cf["gamma"]):
        return "---"
    s = f"${cf['gamma']:+.3f}$"
    t = cf.get("gamma_t_wcb")
    if isinstance(t, (int, float)) and math.isfinite(t):
        s += f" $({t:+.1f})$"
    return s


def _iter_cells(wiv):
    for spec in SPEC_ORDER:
        rr = wiv.get(spec)
        if not rr:
            continue
        yield spec, rr


def build_firststage(wiv):
    # Spec | Sample | N | K | F pooled | F two-way | eff-F(2way) | partial R^2 (2way) | J (p)
    col_fmt = "ll r r r r r c r"
    header = (r"Instruments & Sample & $N$ & $K$ & $F$ pooled & $F$ two-way & eff-$F$ (2way) & "
              r"partial $R^2$ (2way) & $J$ ($p$)")
    body = []
    for spec, rr in _iter_cells(wiv):
        for i, (key, lbl) in enumerate(SUBS):
            r = rr.get(key)
            if not r:
                continue
            fp = r.get("first_stage_strength", {}).get("pooled", {})
            ft = r.get("first_stage_strength", {}).get("twoway", {})
            pv = f"${r['hansen_J_p']:.2f}$" if isinstance(r.get("hansen_J_p"), (int, float)) else "---"
            body.append("    " + " & ".join([
                SPEC_LABEL[spec] if i == 0 else "", lbl,
                f"{r.get('n_obs', 0):,}", str(r.get("n_iv", "")),
                _Ff(fp.get("joint_F")), _Ff(ft.get("joint_F")), _Ff(ft.get("eff_F")),
                _num(ft.get("partial_R2"), 3, signed=False), pv,
            ]) + r" \\")
        body.append(r"    \addlinespace[0.4ex]")
    foot = (r"\textit{Notes:} First stage of the sleepiness model: the QoQ deposit spread "
            r"\texttt{spread\_qoq} (deposit types 4 and 5, the only endogenous regressor) on $K$ "
            r"excluded instruments, clustered by prudential conglomerate. \emph{Pooled} is the first "
            r"stage as estimated (\texttt{run\_pooled\_first\_stage}); \emph{two-way} additionally "
            r"removes entity (bank$\times$type$\times$market) and quarter fixed effects by alternating "
            r"projections---the honest first stage, since the control-function second stage identifies "
            r"$\hat\phi$ within entity net of common time. $F$ = cluster-robust joint $F$ of the excluded "
            r"instruments; eff-$F$ = Montiel-Olea--Pflueger effective $F$. The pooled $F$ is strong and "
            r"the two-way $F$ moderate: the instruments' between/common variation is absorbed, but "
            r"within-entity relevance survives.")
    return _wrap(body, col_fmt, r"Sleepiness First-Stage Instrument Strength (Pooled vs.\ Two-Way FE)",
                 "tab:firststage_sleep", header, foot, 9)


def build_alpha(wiv):
    # Spec | Sample | eff-F(2way) | OLS a | 2SLS a(SE) | LIML a | tF 95% CI | gamma_CF (t)
    col_fmt = "ll r r l r c r"
    header = (r"Instruments & Sample & eff-$F$ & OLS $\hat\alpha$ & 2SLS $\hat\alpha$ (SE) & "
              r"LIML $\hat\alpha$ & tF 95\% CI & $\hat\gamma_{\mathrm{CF}}$ ($t$)")
    body = []
    for spec, rr in _iter_cells(wiv):
        for i, (key, lbl) in enumerate(SUBS):
            r = rr.get(key)
            if not r:
                continue
            ft = r.get("first_stage_strength", {}).get("twoway", {})
            body.append("    " + " & ".join([
                SPEC_LABEL[spec] if i == 0 else "", lbl,
                _Ff(ft.get("eff_F")), _num(r.get("alpha_ols")),
                _alpha_se(r.get("alpha_2sls"), r.get("alpha_se")), _num(r.get("alpha_liml")),
                _tf(r), _gamma_t(r.get("cf_hausman")),
            ]) + r" \\")
        body.append(r"    \addlinespace[0.4ex]")
    foot = (r"\textit{Notes:} \textbf{$\hat\alpha$ is a LINEAR-IV PROJECTION benchmark---2SLS of the "
            r"two-way-FE-demeaned deposit balance on \texttt{spread\_qoq} instrumented by the excluded "
            r"instruments---NOT the structural sleepiness parameter $\phi$}; it shares the first stage "
            r"and exclusion restriction with the control-function estimator, so it diagnoses the "
            r"identification $\hat\phi$ relies on, and it is the only setting where the "
            r"identification-robust tools are valid. The estimator ladder (OLS$\to$2SLS$\to$LIML) gives "
            r"the sign; tF 95\% CI is the $F$-adjusted honest interval of Lee, McCrary, Moreira \& "
            r"Porter (2022). $\hat\gamma_{\mathrm{CF}}$ is the STRUCTURAL control-function coefficient on "
            r"$\hat v\times$lagged deposits in the actual two-way-FE second stage, with its "
            r"wild-cluster-bootstrap $t$: under $H_0:\gamma=0$ this is a valid few-cluster endogeneity "
            r"(Durbin--Wu--Hausman) test; tF is deliberately not applied to a generated-regressor "
            r"coefficient. Deposits are highly concentrated, leaving only $G^\ast\approx5$--$7$ "
            r"effective clusters, so the intervals reflect \emph{precision}, not weak relevance.")
    return _wrap(body, col_fmt,
                 r"Identification-Robust Inference: Linear-IV Projection Benchmark and CF Endogeneity Test",
                 "tab:alpha_weakiv_sleep", header, foot, 8)


def build_phi(wiv):
    # Link | mean phi OLS | mean phi IV | mean|d| | Pearson | Spearman | trend OLS | trend IV
    phi = wiv.get("phi_need_test", {})
    col_fmt = "l r r r r r r r"
    header = (r"Link & mean $\hat\phi$ OLS & mean $\hat\phi$ IV & mean$|\Delta|$ & "
              r"Pearson & Spearman & trend OLS & trend IV")
    body = []
    for link, lbl in (("logit", r"Logit (E3/E4)"), ("sieve", r"Sieve (E7/E8)")):
        s = phi.get(link)
        if not s:
            continue
        body.append("    " + " & ".join([
            lbl, _num(s.get("mean_phi_ols"), 3, signed=False), _num(s.get("mean_phi_iv"), 3, signed=False),
            _num(s.get("mean_abs_delta"), 3, signed=False), _num(s.get("pearson")),
            _num(s.get("spearman")), _num(s.get("trend_ols")), _num(s.get("trend_iv")),
        ]) + r" \\")
    if not body:
        body = [r"    \multicolumn{8}{c}{\textit{phi test not available}} \\"]
    foot = (r"\textit{Notes:} National \emph{deposit-weighted} $\hat\phi_t$ reconstructed under OLS "
            r"(no control function) vs.\ IV\_HausmanFull (control function), spec~12 "
            r"(IV\_HausmanFull\,$\times$\,Tech), on the common sample where the instruments exist. "
            r"$\hat\phi$ is built from the native index $\times$ link via \texttt{phi\_from\_native} "
            r"(the logit and the E7/E8 monotone sieve). A strong negative correlation and a reversed "
            r"trend indicate the control function materially changes the estimated inattention (it "
            r"pulls $\hat\phi$ off the boundary), so it earns its place.")
    return _wrap(body, col_fmt, r"Does the Control Function Move the Estimated Sleepiness $\hat\phi_t$? (Spec.~12)",
                 "tab:phi_need_sleep", header, foot, 8)


def main():
    if not WEAK_IV.exists():
        raise FileNotFoundError(f"weak_iv_sleep.json not found at {WEAK_IV} -- run weak_iv_sleep_analysis.py first.")
    wiv = json.load(open(WEAK_IV, encoding="utf-8"))
    tables = {
        "tab_firststage_sleep.tex": build_firststage(wiv),
        "tab_alpha_weakiv_sleep.tex": build_alpha(wiv),
        "tab_phi_need_sleep.tex": build_phi(wiv),
    }
    for name, tex in tables.items():
        for dest in (TABLES_DIR, DRAFTS_DIR):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / name).write_text(tex, encoding="utf-8")
        print(f"  wrote {name}")
    print("Done. \\input{tab_firststage_sleep.tex} / {tab_alpha_weakiv_sleep.tex} / {tab_phi_need_sleep.tex} "
          "(needs \\usepackage{pdflscape,longtable,booktabs,setspace}).")


if __name__ == "__main__":
    main()
