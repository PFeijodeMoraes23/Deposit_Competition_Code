"""
estimation_timeseries_test.py  (post-relineup, reader-based)
================================================================================
Builds the consolidated sleepiness report from the E1-E9 pipeline outputs. With the
relineup, the "+Time" robustness is now first-class: the time block (time_trend +
gdp_growth_yoy, added to every state block) defines the paired estimators

    Logit         E3 (Base)  vs  E4 (+Time)
    Single-Index  E5 (Base)  vs  E6 (+Time)
    Joint sieve   E7 (Base)  vs  E8 (+Time)

so the time-series test is just the Base-vs-+Time comparison across these pairs.
The report also folds in the single-index LINK comparison (E5 index vs E7 sieve, and
the optional E9 kernel) that used to live in compare_si_links_spec12.

This script READS the pipeline outputs (no re-estimation):
  ESTIMATION_OUTPUT/DEMAND_PREP/est{N}/national_phi_t.csv   (national phi_t, spec 12)
  ESTIMATION_OUTPUT/DEMAND_PREP/est{N}/estimation_results.pkl  (spec 12 AMEs)

Outputs (Drafts/Deposit Competition): Sleepiness_TimeSeries_Test.{md,pdf} + figures.

CLI: python estimation_timeseries_test.py            # build from current pipeline outputs
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle
import subprocess
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

from utils import paths as P
from utils.sleep_links import NonLinearResults  # noqa: F401 (unpickling)

DP = P.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
ROUT = P.PROCESSED / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS = pathlib.Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
                      r"\Open Finance\Open-Finance\Drafts\Deposit Competition")
SPEC_KEY = "IV_HausmanFull x Tech"
PHI_COL = "phi_t_IV_HausmanFull_x_Tech"

# (label, base est, +Time est) pairs
PAIRS = [("Logit", 3, 4), ("Single-Index", 5, 6), ("Joint sieve", 7, 8)]
# single-index link comparison (label, est, color)
LINKS = [("Single-Index (E5)", 5, "#1b9e77"), ("Joint sieve (E7)", 7, "#7570b3"),
         ("Joint kernel (E9)", 9, "#d95f02")]
# (kind, time_block) for re-fitting the link estimators with phi_band -> CI bands
LINK_KIND = {5: ("single_index", False), 7: ("joint_sieve", False), 9: ("joint_kernel", False)}


def _link_band(est_num, refit=False):
    """National phi_t with a 95% wild-cluster-bootstrap band for the single-index link
    estimator E{est_num}, by re-fitting spec 12 with phi_band=True (cached). Returns a
    DataFrame [time_id, phi_t, lo, hi, date] or None on failure."""
    cache = ROUT / f"ts_link_band_est{est_num}.pkl"
    if cache.exists() and not refit:
        band = pickle.load(open(cache, "rb"))
    else:
        try:
            from estimation_2_sleep import build_pooled_data, define_specifications, run_pooled_first_stage
            from utils.sleep_links import fit_nlls_link, fit_single_index, fit_joint_single_index
            kind, tb = LINK_KIND[est_num]
            df = build_pooled_data(time_block=tb)
            _, iv, sb = define_specifications(time_block=tb)
            s_cols = sb["Tech"]
            df, _ = run_pooled_first_stage(df, iv["IV_HausmanFull"], [c for c in s_cols if c != "constant"])
            logit = fit_nlls_link(df, s_cols, has_cf=True, link="logit", loss="cauchy")
            if kind == "single_index":
                res = fit_single_index(df, s_cols, True, logit, degree=3, phi_band=True)
            else:
                idx = [c for c in s_cols if c != "constant"]
                init = np.array([float(logit.params_native.get(f"interaction_{c}", 0.0)) for c in idx])
                link = "sieve" if kind == "joint_sieve" else "kernel"
                res = fit_joint_single_index(df, s_cols, True, link=link, loss="robust", init_theta=init,
                                             n_starts=(1 if link == "kernel" else 2), boot_B=999,
                                             boot_scheme="webb", seed=0, phi_band=True, label=f"ts/E{est_num}")
            band = res.phi_t_boot
            pickle.dump(band, open(cache, "wb"))
        except Exception as e:
            print(f"  [band est{est_num}] failed: {e}")
            return None
    band = band.copy()
    band["date"] = pd.PeriodIndex(band["time_id"].str.replace("Q", "Q"), freq="Q").to_timestamp()
    return band.sort_values("date").reset_index(drop=True)
TIME_VARS = ["time_trend", "gdp_growth_yoy"]
CLEAN = {"interaction_time_trend": "Time trend (years)", "interaction_gdp_growth_yoy": "GDP growth (YoY)",
         "interaction_pix_exists": "Pix available", "interaction_gdp_per_capita": "GDP p.c.",
         "interaction_cadunico_families_per1000": "CadUnico", "interaction_fraction_65plus": "Fraction 65+",
         "interaction_fraction_young": "Fraction young", "interaction_risk_free_qoq_lag": "Lagged Selic",
         "interaction_connections_per100": "Broadband"}


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def load_phi_t(est_num):
    f = DP / f"est{est_num}" / "national_phi_t.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if PHI_COL not in df.columns:
        return None
    out = df[["year_quarter", PHI_COL]].rename(columns={PHI_COL: "phi_t"}).dropna()
    out["date"] = pd.PeriodIndex(out["year_quarter"].str.replace("Q", "Q"), freq="Q").to_timestamp()
    return out.sort_values("date").reset_index(drop=True)


def load_ss(est_num):
    f = DP / f"est{est_num}" / "estimation_results.pkl"
    if not f.exists():
        return None
    try:
        r = pickle.load(open(f, "rb")).get(SPEC_KEY, {})
        return r.get("second_stage")
    except Exception:
        return None


# Full stage-2 comparison: same lineup/rows as export_analyze_spec12.py's stage-2 table.
S2_ESTS = [(1, "Local"), (2, "Pooled"), (5, "Single-Idx"),
           (6, "Single-Idx +T"), (7, "Joint Sieve"), (8, "Joint Sieve +T")]
S2_ROWS = [("nr_lagged_dep", "Constant"),
           ("interaction_gdp_per_capita", "GDP p.c."),
           ("interaction_cadunico_families_per1000", "CadUnico"),
           ("interaction_fraction_65plus", "Fraction 65+"),
           ("interaction_fraction_young", "Fraction young"),
           ("interaction_risk_free_qoq_lag", "Lagged Selic"),
           ("interaction_connections_per100", "Broadband"),
           ("interaction_pix_exists", "Pix available"),
           ("interaction_gdp_growth_yoy", "GDP growth (YoY)")]


def _ss_clusters(ss):
    """Nominal cluster count G: from the linear cov_kwds groups, else NonLinearResults.G_nominal."""
    ck = getattr(ss, "cov_kwds", None)
    if ck and ck.get("groups", None) is not None:
        g = ck["groups"]
        return str(g.nunique() if hasattr(g, "nunique") else len(set(g)))
    gn = getattr(ss, "G_nominal", np.nan)
    return str(int(gn)) if pd.notna(gn) else "—"


def stage2_comparison_md():
    """Pandoc-native markdown table of the full stage-2 AME comparison across the six
    estimators, wrapped in raw-LaTeX \\begin{landscape}...\\end{landscape} for preview."""
    ssn = {n: load_ss(n) for n, _ in S2_ESTS}
    if all(ss is None for ss in ssn.values()):
        return []

    header = "| Variable | " + " | ".join(lbl for _, lbl in S2_ESTS) + " |"
    sep = "|" + "|".join(["---"] * (len(S2_ESTS) + 1)) + "|"
    L = ["```{=latex}", "\\begin{landscape}", "```", "",
         "## Stage-2 sleepiness coefficients / AMEs — full comparison (spec 12)", "",
         "Linear sleepiness coefficients (Local, Pooled) and average marginal effects "
         "(Single-Index, Joint Sieve); wild-cluster-bootstrap SE in parentheses "
         "(*** p<0.01, ** p<0.05, * p<0.1), applied uniformly to every column. The "
         "**Mean $\\hat\\phi$** row is the implied national sleepiness level, comparable across "
         "all six columns. Preview of the canonical `est1-3_spec12_stage2_comparison.tex` "
         "(where V_Main numbers the columns via the estimation-strategy enumeration).", "",
         header, sep]

    for key, lbl in S2_ROWS:
        cells = []
        for n, _ in S2_ESTS:
            ss = ssn[n]
            if ss is not None and key in getattr(ss, "params", pd.Series(dtype=float)).index:
                a = ss.params[key]; se = ss.bse.get(key, np.nan); p = ss.pvalues.get(key, np.nan)
                cells.append(f"{a:+.4f}{_stars(p)} ({se:.4f})")
            else:
                cells.append("—")
        L.append(f"| {lbl} | " + " | ".join(cells) + " |")

    # diagnostic rows
    mean_cells, nobs_cells, r2_cells, g_cells = [], [], [], []
    for n, _ in S2_ESTS:
        ss = ssn[n]
        pt = load_phi_t(n)
        mean_cells.append(f"{pt['phi_t'].mean():.3f}" if pt is not None else "—")
        if ss is None:
            nobs_cells.append("—"); r2_cells.append("—"); g_cells.append("—")
            continue
        nobs = getattr(ss, "nobs", np.nan); r2 = getattr(ss, "rsquared", np.nan)
        nobs_cells.append(f"{nobs:,.0f}" if pd.notna(nobs) else "—")
        r2_cells.append(f"{r2:.3f}" if pd.notna(r2) else "—")
        g_cells.append(_ss_clusters(ss))

    L.append("| **Mean $\\hat\\phi$ (level)** | " + " | ".join(mean_cells) + " |")
    L.append("| Observations | " + " | ".join(nobs_cells) + " |")
    L.append("| $R^2$ | " + " | ".join(r2_cells) + " |")
    L.append("| Clusters ($G$) | " + " | ".join(g_cells) + " |")
    L += ["", "```{=latex}", "\\end{landscape}", "```", ""]
    return L


def cluster_imbalance_md():
    """Cluster-imbalance / deposit-concentration two-panel table (the WCB
    justification), as a pandoc-native markdown table computed on the E7 second-stage
    sample via desc_3 (single source of the numbers)."""
    try:
        from desc_3 import load_sample, build_stats
        df, _ = load_sample(7)
        st, _ = build_stats(df, top_n=5)
    except Exception as e:
        print(f"  [cluster_imbalance] skipped: {e}")
        return []

    def pct(x):
        return f"{100 * x:.1f}%"
    L = ["## Cluster imbalance and deposit concentration (spec-12 second-stage sample)", "",
         "Why the wild cluster bootstrap: the conglomerate clusters are few in *effective* terms "
         "and wildly unequal in size. On the spec-12 second-stage sample "
         f"({int(st['total_obs']):,} observations, {st['G_nominal']} conglomerates) the "
         f"**{st['G_nominal']} nominal clusters collapse to $G^{{*}}\\approx{st['G_star']:.0f}$ "
         f"effective** (Carter et al. 2017), with a coefficient of variation of cluster sizes of "
         f"{st['cv']:.1f}; that is precisely the regime where the cluster-robust / Delta-method "
         "$t$-test over-rejects (MacKinnon-Webb 2017).", "",
         "**Panel A --- cluster structure**", "",
         "| Metric | Value |", "|---|---|",
         f"| Clusters (conglomerates), $G$ | {st['G_nominal']:,} |",
         f"| Effective clusters, $G^{{*}}=G/(1+\\mathrm{{CV}}^2)$ | {st['G_star']:.1f} |",
         f"| Total observations | {int(st['total_obs']):,} |",
         f"| Obs. per cluster: mean / median / max | {st['mean_size']:,.0f} / {st['median_size']:,.0f} / {st['max_size']:,.0f} |",
         f"| CV of cluster sizes | {st['cv']:.2f} |",
         f"| Deposit HHI ($1/$HHI) | {st['hhi']:.3f} ({st['inv_hhi']:.1f}) |", "",
         "**Panel B --- five largest conglomerates by national deposit share**", "",
         "| Rank | Conglomerate | Dep. share | Cumulative | Obs. share |",
         "|---|---|---|---|---|"]
    for i, r in enumerate(st["top_n"], 1):
        L.append(f"| {i} | {r['name']} | {pct(r['dep_share'])} | {pct(r['cum_share'])} | {pct(r['obs_share'])} |")
    L.append("")
    return L


def fig_pair(name, eb, et, path):
    pb, pt = load_phi_t(eb), load_phi_t(et)
    if pb is None or pt is None:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(pb["date"], pb["phi_t"], color="#1b9e77", lw=2, label=f"E{eb} Base")
    ax.plot(pt["date"], pt["phi_t"], color="#d95f02", lw=2, ls="--", label=f"E{et} + Time")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower right")
    ax.set_ylabel(r"National $\hat{\phi}_t$"); ax.set_title(f"{name}: Base (E{eb}) vs + Time block (E{et}), spec 12")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return True


def fig_links(path, bands=True, refit=False):
    """Single-index link comparison. With bands=True, re-fit E5/E7 (and E9 if its
    pipeline result exists) with phi_band -> 95% bootstrap bands; else point paths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for label, est, color in LINKS:
        band = None
        if bands and est in LINK_KIND:
            if est == 9 and not (DP / "est9" / "estimation_results.pkl").exists() and not refit:
                band = None   # don't trigger the slow kernel re-fit unless E9 was run
            else:
                band = _link_band(est, refit=refit)
        if band is not None:
            ax.plot(band["date"], band["phi_t"], color=color, lw=2, label=label)
            ax.fill_between(band["date"], band["lo"], band["hi"], color=color, alpha=0.18, lw=0)
            rows.append((label, band["phi_t"].mean(), band["phi_t"].min(), band["phi_t"].max(), True))
        else:
            p = load_phi_t(est)
            if p is None:
                continue
            ax.plot(p["date"], p["phi_t"], color=color, lw=2, ls="--", label=f"{label} (point)")
            rows.append((label, p["phi_t"].mean(), p["phi_t"].min(), p["phi_t"].max(), False))
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower right")
    ax.set_ylabel(r"National $\hat{\phi}_t$")
    ax.set_title("Single-index LINK comparison (spec 12, 95% wild-cluster-bootstrap bands)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return rows


def build(bands=True, refit=False):
    L = ["# Sleepiness report — time block (E3/E4, E5/E6, E7/E8) and single-index links", ""]
    L += ["This report documents the **sleepiness estimation routines** (E1–E8), their estimation "
          "logic, and the **two-way fixed-effects** structural change. The methodology in the next "
          "four sections reflects the **current** specification (additive entity + quarter FE). "
          "*The $\\hat\\phi_t$ figures and AME tables below reflect the completed **two-way "
          "fixed-effects** run (E1–E8; additive entity + quarter FE); the single-index link "
          "comparison carries 95% score/multiplier wild-cluster-bootstrap CI bands (computed in the "
          "main estimation routine for spec 12).*", "",
          "Under the relineup the **+Time** estimators (E4/E6/E8) now append only `gdp_growth_yoy` to "
          "the index — the pure-time `time_trend` was dropped because it is collinear with the quarter "
          "fixed effect — so the **Base vs +Time** pairs are:", "",
          "| Pair | Base | +Time |", "|---|---|---|",
          "| Logit | E3 | **E4** |", "| Single-Index | E5 | **E6** |", "| Joint sieve | E7 | **E8** |", ""]

    # =====================  METHODOLOGY  =====================
    L += [
      "## Estimation framework", "",
      "The sleepiness function $\\phi_{mt}=\\phi(\\mathbf S_{mt})\\in[0,1]$ is the share of "
      "depositors in market $m$ at quarter $t$ who do **not** re-optimize ('asleep'), so their "
      "balances carry forward; it is identified from the autocorrelation of deposits. Writing "
      "$\\widetilde D_{jkmt-1}=(1+r_{t-1}-\\rho_{jkmt-1})\\,\\mathrm{Dep}^{\\mathrm B}_{jkmt-1}$ for "
      "the interest-accrued lagged stock (the **carry term** $Z$, `nr_lagged_dep`), the structural "
      "deposit equation (V\\_Main eq. 13) is", "",
      "$$\\mathrm{Dep}^{\\mathrm B}_{jkmt} = \\phi(\\mathbf S_{mt})\\,\\widetilde D_{jkmt-1} "
      "+ H(\\hat v_{jkmt},\\mathrm{Dep}_{jkmt-1}) + \\alpha_{jkm} + \\delta_t + u_{jkmt},$$", "",
      "with $\\phi(\\mathbf S)=G(\\mathbf S'\\theta)$ for a link $G$. The pieces:", "",
      "- **Carry term $Z=\\widetilde D_{t-1}$** — $\\phi$ loads on it (high $\\phi$ = sticky deposits).",
      "- **Control function $H$** (`v_hat_x_lagged_dep`). OLS of eq. 13 is biased because lagged "
      "deposits co-move with omitted active demand (vertical differentiation). A first stage "
      "regresses spreads on cost-shifter instruments $\\mathbf Z^{\\mathrm{Sleep}}$ (eq. 12); the "
      "residual $\\hat v$ — the **latent demand** — interacted with the lagged stock enters the "
      "second stage and purges the bias transmitted through $\\rho_{t-1}$ (Newey–Powell–Vella 1999).",
      "- **Fixed effects $\\alpha_{jkm}+\\delta_t$** — see the next section.",
      "- **State vector $\\mathbf S$** — a constant, `pix_exists`, and (by specification) GDP per "
      "capita, CadÚnico family density, age structure, the lagged risk-free rate and broadband "
      "connections; spec 12 (the BLP input) is the 'Tech' block under the Hausman-style instruments.", "",
      "National sleepiness is the population-weighted average "
      "$\\hat\\phi_t=\\sum_m M_{mt}\\hat\\phi_{mt}/M_t$.", "",
      "## Fixed effects: the two-way change", "",
      "**What changed.** Eq. 13 originally carried only **bank $\\times$ type $\\times$ market** "
      "effects $\\alpha_{jkm}$ — a purely cross-sectional unit effect (12{,}255 entities, each "
      "spanning about 36 of 51 quarters, *no time dimension*). We add a **quarter** fixed effect "
      "$\\delta_t$, giving an **additive two-way** structure $\\alpha_{jkm}+\\delta_t$ across all E1–E8.", "",
      "**Why.** The control function removes the endogeneity transmitted through $\\rho_{t-1}$; the "
      "fixed effects 'absorb all remaining bias entering through $\\mathrm{Dep}_{t-1}$' (V\\_Main line "
      "390) — i.e. the latent demand $v$. The entity FE absorb its time-invariant, cross-sectional "
      "part; but if $v$ has a **time-varying aggregate** component correlated with lagged deposits "
      "(the 2020 COVID deposit surge is exactly such a shock), the entity FE miss it and OVB remains. "
      "$\\delta_t$ absorbs it — the same OVB/latent-demand argument, now on the **time** margin.", "",
      "**How — and why not just demean $\\mathbf S$.** The FE are **additive, outside the link, and "
      "not interacted with $\\widetilde D_{t-1}$**. For a nonlinear link one **cannot** remove them by "
      "demeaning the state space, since $G(\\mathbf S-\\bar{\\mathbf S})\\neq G(\\mathbf S)-"
      "\\overline{G(\\mathbf S)}$ (Frisch–Waugh–Lovell holds only for the linear model). Instead we "
      "apply the link to the **raw** $\\mathbf S$, form the additive prediction "
      "$G(\\mathbf S'\\theta)\\,Z+H$, and concentrate out $\\alpha_{jkm}+\\delta_t$ by "
      "**within-demeaning the prediction (and the outcome)** — two-way via alternating projections "
      "(Gaure 2013). Because the FE enter additively and are *differenced out* (not estimated by "
      "nonlinear MLE), there is **no incidental-parameters bias**; the wild cluster bootstrap inherits "
      "the two-way transform automatically (through the demeaned residual/Jacobian).", "",
      "**Index regressors under the time FE.** With $\\delta_t$ absorbing aggregate time, `time_trend` "
      "(a pure-time ramp, zero within-quarter variation) is collinear with the quarter FE and is "
      "**dropped**; the '+Time' block reduces to `gdp_growth_yoy` (a genuine entity-time covariate). "
      "`pix_exists` is **kept** as the Pix treatment. Empirically the two-way FE **flattens** "
      "$\\hat\\phi_t$ — this is the **bias correction**: the aggregate variation it removes (notably "
      "the post-2020 climb toward the ceiling) was time-varying latent demand previously misattributed "
      "to rising sleepiness. The logit (E3/E4) still saturates regardless — that is a functional-form "
      "pathology of the unbounded index, orthogonal to the FE, so the bounded **E7** is canonical.", "",
      "## Estimation routines (E1–E8)", "",
      "All share eq. 13, the control function, the two-way FE and the wild-cluster-bootstrap "
      "inference; they differ in the **functional form of $\\phi$**, the **data coverage** and the "
      "**estimation method**. The **robust (Cauchy)** fit feeds $\\phi$; a plain-LS variant is "
      "reported alongside.", "",
      "- **E1 — Local-linear** (`estimation_1_sleep.py`). Linear "
      "$\\phi=\\boldsymbol\\Upsilon'\\mathbf S$, **B-firms only** (digital banks excluded). Two-way "
      "within-OLS of the demeaned interacted design $[\\,S_k\\widetilde D_{t-1}\\,]_k$ on demeaned "
      "deposits; **wild cluster bootstrap** SE at the conglomerate level — the same score/multiplier "
      "scheme as every other column (see the AME section) — with the Carter–Schnepel–Steigerwald "
      "$G^*$ reported as a diagnostic. Unconstrained, so $\\phi$ may exit $[0,1]$.",
      "- **E2 — Pooled-linear** (`estimation_2_sleep.py`). As E1 but **pooled** over B and D firms; "
      "hosts the shared data build.",
      "- **E3 — Logit.** $\\phi=\\Lambda(\\mathbf S'\\theta)$, bounded by construction. Estimated by "
      "**nonlinear least squares** (trust-region, Cauchy loss) on the two-way-demeaned residual "
      "$D-\\Lambda(\\mathbf S'\\theta)Z-\\gamma(\\hat vZ)$; analytic AMEs, wild-cluster-bootstrap SE.",
      "- **E4 — Logit + Time.** E3 with `gdp_growth_yoy` appended to the index.",
      "- **E5 — Single-Index.** $\\phi=G(\\mathbf S'\\theta)$ with a **monotone cubic B-spline "
      "(sieve)** link. The index **direction $\\theta$ is taken from the logit** (E3); $G$ is then fit "
      "by OLS of the two-way-demeaned $[\\,B_d(\\mathbf S'\\theta)\\,\\widetilde D_{t-1}\\,]$ basis, "
      "with the average-derivative AME read off the spline. Bounded and smooth, but inference is "
      "conditional on $\\theta$.",
      "- **E6 — Single-Index + Time.** E5 with `gdp_growth_yoy`.",
      "- **E7 — Joint sieve.** Ichimura (1993) **semiparametric least squares**: estimate the index "
      "direction $\\theta$ ($\\lVert\\theta\\rVert=1$) **and** the link $G$ **jointly**, minimizing "
      "the two-way-demeaned SSR with $G$ profiled out at each $\\theta$. $G$ is a **monotone I-spline** "
      "(cumulative cubic B-spline ramps with non-negative coefficients — monotone and in $[0,1]$ by "
      "construction; Ramsay 1988). The $\\theta$-search is Nelder–Mead (1965) multistart over the unit sphere "
      "(warm-started from the logit direction); each evaluation re-fits the profiled link by "
      "bounded-variable least squares. This removes all three single-index caveats (direction "
      "consistency, fixed-degree link, SEs that ignore $\\hat\\theta$) and is the **BLP-input** "
      "estimator; SEs by score/multiplier wild cluster bootstrap on $\\theta\\to$AMEs.",
      "- **E8 — Joint sieve + Time.** E7 with `gdp_growth_yoy`.",
      "- *(E9 — Joint kernel: local-linear link, monotonised by rearrangement; deferred, not in the "
      "default lineup.)*", "",
      "## Computational implementation (Julia)", "",
      "The two-way demean runs **per $\\theta$-evaluation** inside the joint sieve (E7/E8), making "
      "those the wall-clock bottleneck. We moved the **$\\theta$-search only** to Julia "
      "(`sleep_joint_sieve.jl`): Python keeps the data build, the first stage, the **final link refit "
      "at $\\hat\\theta$** (with an exact demean), $\\phi$, AMEs and the bootstrap — Julia returns "
      "$\\hat\\theta$. The hot loop is **allocation-free**: the profiled monotone-sieve BVLS is solved "
      "from $K\\times K$ Gram cross-products (the control-function coefficient partialled out by FWL, "
      "then Lawson–Hanson (1974) NNLS in the fast normal-equations form of Bro–De Jong (1997)), with "
      "in-place two-way demeaning, a single "
      "sorted buffer for the knot quantiles, and a buffered Cauchy median/MAD. A **subsample-$\\theta$** "
      "step (estimate the low-dimensional direction on a fraction of entities, then refit the link on "
      "the full sample) adds a further multiplicative speed-up. The Julia $\\theta$-search reproduces "
      "the Python optimum (validated by direction cosine and an equal-or-lower SLS objective).", ""]
    # =====================  END METHODOLOGY  =====================

    # ---- time-block coefficient table (E4/E6/E8) ----
    L += ["## Does the time block matter? (spec 12)", "",
          "Average marginal effects of the two time variables (wild cluster bootstrap SE in "
          "parentheses; *** p<0.01, ** p<0.05, * p<0.1), from the +Time estimators.", "",
          "| Time variable | E4 (Logit+Time) | E6 (Single-Index+Time) | E8 (Joint sieve+Time) |",
          "|---|---|---|---|"]
    ssn = {n: load_ss(n) for n in (4, 6, 8)}
    for v in TIME_VARS:
        key = f"interaction_{v}"
        cells = []
        for n in (4, 6, 8):
            ss = ssn[n]
            if ss is not None and key in ss.params.index:
                a, se, p = ss.params[key], ss.bse.get(key, np.nan), ss.pvalues.get(key, np.nan)
                cells.append(f"{a:+.4f}{_stars(p)} ({se:.4f})")
            else:
                cells.append("—")
        L.append(f"| {CLEAN[key]} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- full stage-2 comparison table (landscaped preview of the V_Main table) ----
    L += stage2_comparison_md()

    # ---- cluster-imbalance / deposit-concentration table (WCB justification) ----
    L += cluster_imbalance_md()

    # ---- phi_t base vs +time per pair ----
    L += ["## National $\\hat\\phi_t$: Base vs +Time", ""]
    for name, eb, et in PAIRS:
        png = f"ts_pair_{name.lower().replace(' ', '').replace('-', '')}.png"
        if fig_pair(name, eb, et, DRAFTS / png):
            pb, pt = load_phi_t(eb), load_phi_t(et)
            L.append(f"**{name}** — Base (E{eb}) mean $\\hat\\phi_t$ = {pb['phi_t'].mean():.3f}; "
                     f"+Time (E{et}) mean = {pt['phi_t'].mean():.3f}.")
            L.append("")
            L.append(f"![{name}: Base vs +Time]({png}){{width=85%}}")
            L.append("")

    # ---- single-index link comparison ----
    L += ["## Single-index link comparison (spec 12)", "",
          "How sensitive is the implied sleepiness to the link family — the conditional cubic-sieve "
          "single index (E5), the **jointly**-estimated monotone sieve (E7), and the optional joint "
          "**kernel** (E9)?", ""]
    rows = fig_links(DRAFTS / "ts_links.png", bands=bands, refit=refit)
    if rows:
        L.append("![Single-index link comparison](ts_links.png){width=90%}")
        L.append("")
        L.append("| Estimator | mean $\\hat\\phi_t$ | range | inference |")
        L.append("|---|---|---|---|")
        for label, mean, lo, hi, has_band in rows:
            L.append(f"| {label} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | "
                     f"{'95% bootstrap band' if has_band else 'point only'} |")
        L.append("")
        L.append("*Bands are 95% score/multiplier wild-cluster-bootstrap intervals on the national "
                 "$\\hat\\phi_t$ (perturbing the cluster-summed influence functions; the link held "
                 "fixed). Recall the robust **kernel** link can degenerate to the $[0,1]$ ceiling, so "
                 "for E9 prefer the LS link.*")
        L.append("")

    L += ["## Average marginal effects (AME): implementation", "",
          "Every AME table above (and in the per-estimator export tables) reports the sample-average "
          "effect of a state variable on the sleepiness probability $\\phi=G(\\mathbf S'\\theta)$, "
          "computed **analytically** and given inference by a **wild cluster bootstrap**. (This merges "
          "and updates the old standalone AME notes.)", "",
          "**The effects (analytical).** For each regressor $S_k$:", "",
          "- **Dummies** (e.g. Pix): the exact discrete difference "
          "$\\frac1n\\sum_i\\big[G(\\cdot,S_k{=}1)-G(\\cdot,S_k{=}0)\\big]$, bounded in $[-1,1]$.",
          "- **Continuous regressors**: the exact derivative averaged, "
          "$\\frac1n\\sum_i G'(\\mathbf S_i'\\theta)\\,\\theta_k$ — for the logit $G'=P(1-P)$; for the "
          "single index $G'$ is the cubic-sieve / I-spline / kernel link slope (average-derivative AME).",
          "",
          "**The standard errors (wild cluster bootstrap).** The SE printed beside each AME is the SE "
          "*of the AME itself*, not of the underlying index coefficients: the bootstrap perturbs the "
          "cluster-summed influence functions of $\\hat\\theta$ and re-evaluates the AME map "
          "$g(\\hat\\theta)$ on every draw, so the reported figure is the sampling standard deviation of "
          "the average marginal effect (the average-derivative estimand for continuous regressors; "
          "Powell, Stock & Stoker 1989; on computing marginal-effect SEs via delta method vs. bootstrap, "
          "Dowd, Greene & Norton 2014). This supersedes the old "
          "delta-method / numerical-Jacobian SEs ($\\mathrm{Cov}_{AME}\\approx J\\,\\mathrm{Cov}(\\hat"
          "\\theta)\\,J'$ with a finite-difference $J$, which was NOT cluster-robust). Inference is now "
          "a **score/multiplier wild cluster bootstrap** at the conglomerate level "
          "(Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017): the cluster-summed influence functions "
          "are perturbed by wild weights (Webb 6-point, $B=999$) and the AMEs recomputed — no refit per "
          "draw. **The same wild cluster bootstrap is now applied to the linear estimators (E1/E2) "
          "too** — for the linear OLS columns the cluster influence functions are "
          "$(\\mathbf X'\\mathbf X)^{-1}\\sum_{i\\in g}\\mathbf X_i\\hat u_i$ — so **every column of the "
          "comparison tables shares one inference scheme**, with $B=999$ Webb weights at the "
          "conglomerate level. (Imbens-Kolesar's (2016) CR2 bias-reduced linearisation is defined "
          "through the OLS hat matrix, so it is available for the linear columns but does **not** "
          "extend to the nonlinear / profiled-link estimators, which have no hat matrix; we therefore "
          "use the WCB throughout rather than mixing schemes.) The Carter-Schnepel-Steigerwald (2017) "
          "effective-cluster count $G^*$ ($\\approx 9$ here) is reported across all columns as a "
          "diagnostic only.", "",
          "**Reporting-only.** AMEs never enter $\\phi$ — $\\phi$ is always built from the native index "
          "$\\times$ link (`phi_from_native`); the AMEs are for the tables only.", "",
          "### Weak identification under two-way FE: the Pix dummy", "",
          "Two-way (entity + quarter) FE absorb any index regressor whose variation is mostly a "
          "persistent cross-sectional **level** plus a smooth **national trend**; identification of its "
          "sleepiness loading then comes only from the residual **within-market, within-quarter** "
          "variation. `pix_exists` is the extreme case: Pix launched nationally (Nov 2020), so the dummy "
          "flips at essentially the same quarter for every market — it is **collinear with the quarter "
          "FE** and has almost no within-FE variation left.", "",
          "*Evidence — share of each index regressor's variance absorbed by the entity + quarter FE on "
          "the spec-12 estimation panel ($N\\approx438$k; absorbed "
          "$R^2 = 1-\\operatorname{Var}(\\text{resid})/\\operatorname{Var}(\\text{raw})$, and the "
          "surviving residual sd as a fraction of the raw sd):*", "",
          "| Index regressor | FE-absorbed $R^2$ | resid sd / raw sd |",
          "|---|---|---|",
          "| `pix_exists` | **1.0000** | **0.000** |",
          "| Lagged Selic | 0.9996 | 0.020 |",
          "| Fraction 65+ | 0.9899 | 0.100 |",
          "| Fraction young | 0.9889 | 0.105 |",
          "| CadUnico | 0.9671 | 0.181 |",
          "| GDP p.c. | 0.7542 | 0.496 |",
          "| Broadband | 0.6658 | 0.578 |", "",
          "`pix_exists` is the **only** regressor with essentially zero residual variation. With its "
          "loading thereby unidentified, the **unnormalised** logit index direction that E5/E6 inherit "
          "hands it a runaway coefficient — $\\theta_{\\text{pix}}\\approx 300$ in E5 and $45$ in E6, "
          "versus $0.05$ (E7) and $0.97$ (E8), where the joint estimators' unit-norm constraint "
          "$\\lVert\\theta\\rVert=1$ caps it. Before the dummy-AME fix this inflated the "
          "**continuous-style** Pix effect $\\theta_{\\text{pix}}\\cdot\\overline{G'}$ to an impossible "
          "$+6.9$ (E5) / $+1.2$ (E6) — outside the $[-1,1]$ bound a probability change must respect. The "
          "corrected **discrete-difference** AME $\\tfrac1n\\sum_i[G(\\cdot,\\text{pix}{=}1)-"
          "G(\\cdot,\\text{pix}{=}0)]$ is bounded and sensible ($\\approx +0.10$ in E5/E6, "
          "$+0.001$–$0.005$ in E7/E8). The **caveat stands**, however: the Pix effect is weakly "
          "identified under two-way FE, and the single-index estimators (E5/E6), inheriting the large "
          "logit Pix direction, report a larger Pix AME than the unit-norm joint sieve (E7/E8, the BLP "
          "input). Read the Pix row accordingly — the joint-sieve estimate is the conservative one.", "",
          "## References", "",
          "Bro, R., & De Jong, S. (1997). A fast non-negativity-constrained least squares "
          "algorithm. *Journal of Chemometrics* 11(5): 393–401. "
          "[normal-equations form of the NNLS used for the profiled monotone-link solver]", "",
          "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements "
          "for inference with clustered errors. *Review of Economics and Statistics* 90(3): 414–427. "
          "[wild cluster bootstrap]", "",
          "Carter, A. V., Schnepel, K. T., & Steigerwald, D. G. (2017). Asymptotic behavior of a "
          "t-test robust to cluster heterogeneity. *Review of Economics and Statistics* 99(4): 698–709. "
          "[effective number of clusters $G^*$]", "",
          "Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2009). Improving point and interval "
          "estimators of monotone functions by rearrangement. *Biometrika* 96(3): 559–575. "
          "[ex-post monotonisation of the kernel link]", "",
          "Dowd, B. E., Greene, W. H., & Norton, E. C. (2014). Computation of standard errors. "
          "*Health Services Research* 49(2): 731–750. "
          "[standard errors for marginal/incremental effects: delta method vs. bootstrap]", "",
          "Egan, M., Hortaçsu, A., & Matvos, G. (2017). Deposit competition and financial fragility: "
          "evidence from the US banking sector. *American Economic Review* 107(1): 169–216. "
          "[foundational deposit-demand model; the dynamic depositor-sleepiness extension followed "
          "here is cited as `egan2025dynamic` in V\\_Main — fill the exact working-paper entry from "
          "References.bib]", "",
          "Gaure, S. (2013). OLS with multiple high-dimensional category variables. *Computational "
          "Statistics & Data Analysis* 66: 8–18. [two-way fixed effects by alternating projections]", "",
          "Greene, W. H. (2018). *Econometric Analysis*, 8th ed. Pearson, New York. "
          "[textbook treatment of marginal effects in nonlinear models and their delta-method SEs]", "",
          "Härdle, W., & Stoker, T. M. (1989). Investigating smooth multiple regression by the "
          "method of average derivatives. *Journal of the American Statistical Association* 84(408): "
          "986–995. [average-derivative estimation — the continuous-regressor AME]", "",
          "Ichimura, H. (1993). Semiparametric least squares (SLS) and weighted SLS estimation of "
          "single-index models. *Journal of Econometrics* 58(1–2): 71–120. "
          "[joint single-index SLS — E7/E8]", "",
          "Lawson, C. L., & Hanson, R. J. (1974). *Solving Least Squares Problems*. Prentice-Hall, "
          "Englewood Cliffs, NJ (reprinted as SIAM Classics in Applied Mathematics 15, 1995). "
          "[non-negative least squares (NNLS) active-set algorithm]", "",
          "MacKinnon, J. G., & Webb, M. D. (2017). Wild bootstrap inference for wildly different "
          "cluster sizes. *Journal of Applied Econometrics* 32(2): 233–254. "
          "[Webb 6-point weights for imbalanced clusters]", "",
          "Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. "
          "*The Computer Journal* 7(4): 308–313. [derivative-free optimiser for the joint-sieve "
          "$\\theta$-search]", "",
          "Newey, W. K., Powell, J. L., & Vella, F. (1999). Nonparametric estimation of triangular "
          "simultaneous equations models. *Econometrica* 67(3): 565–603. [control-function approach]", "",
          "Powell, J. L., Stock, J. H., & Stoker, T. M. (1989). Semiparametric estimation of index "
          "coefficients. *Econometrica* 57(6): 1403–1430. "
          "[average-derivative average marginal effect for single-index models]", "",
          "Ramsay, J. O. (1988). Monotone regression splines in action. *Statistical Science* 3(4): "
          "425–441. [I-splines — the monotone sieve link]", "",
          "Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. "
          "MIT Press. [average partial/marginal effects (APE/AME) — the reported estimand and its "
          "inference]", ""]

    out_md = DRAFTS / "Sleepiness_TimeSeries_Test.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] report -> {out_md}")
    return out_md


def compile_pdf(md):
    try:
        r = subprocess.run(["pandoc", md.name, "-o", "_ts_build.pdf", "--pdf-engine=xelatex",
                            "-V", "geometry:margin=1in",
                            "-V", "header-includes=\\usepackage{pdflscape}"],
                           cwd=str(DRAFTS), capture_output=True, text=True)
        built = DRAFTS / "_ts_build.pdf"
        if built.exists() and built.stat().st_size > 0:
            os.replace(built, DRAFTS / "Sleepiness_TimeSeries_Test.pdf")
            print("[OK] PDF compiled")
        else:
            print("[!] PDF compile failed:\n", r.stderr[-800:])
    except Exception as e:
        print(f"[!] PDF compile error: {e}")


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="Build the sleepiness report from the E1-E9 pipeline outputs.")
    pa.add_argument("--no-bands", action="store_true",
                    help="point phi_t paths only for the link comparison (skip the phi_band re-fits)")
    pa.add_argument("--refit-bands", action="store_true",
                    help="force re-fit of the single-index link bands (ignore the cache)")
    a = pa.parse_args()
    md = build(bands=not a.no_bands, refit=a.refit_bands)
    compile_pdf(md)
