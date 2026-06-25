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
    L += ["With the estimator relineup, the **time block** (`time_trend` = years since the sample "
          "start; `gdp_growth_yoy` = within-entity YoY growth of GDP per capita, both interacted with "
          "$\\widetilde D_{t-1}$ and added to every state block) defines first-class **+Time** "
          "estimators, so the old \"does $\\phi$ carry an unmodelled time component?\" test is now the "
          "**Base vs +Time** comparison of three paired estimators:", "",
          "| Pair | Base | +Time |", "|---|---|---|",
          "| Logit | E3 | **E4** |", "| Single-Index | E5 | **E6** |", "| Joint sieve | E7 | **E8** |", ""]

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

    # ---- phi_t base vs +time per pair ----
    L += ["## National $\\hat\\phi_t$: Base vs +Time", ""]
    for name, eb, et in PAIRS:
        png = f"ts_pair_{name.lower().replace(' ', '').replace('-', '')}.png"
        if fig_pair(name, eb, et, DRAFTS / png):
            pb, pt = load_phi_t(eb), load_phi_t(et)
            L.append(f"**{name}** — Base (E{eb}) mean $\\hat\\phi_t$ = {pb['phi_t'].mean():.3f}; "
                     f"+Time (E{et}) mean = {pt['phi_t'].mean():.3f}.")
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

    L += ["## References", "",
          "- Ichimura, H. (1993). Semiparametric least squares estimation of single-index models. "
          "*J. Econometrics* 58: 71–120.",
          "- Cameron, Gelbach & Miller (2008); MacKinnon & Webb (2017) — wild cluster bootstrap.",
          "- Carter, Schnepel & Steigerwald (2017) — effective number of clusters.",
          "- Chernozhukov, Fernández-Val & Galichon (2009) — rearrangement (kernel link).", ""]

    out_md = DRAFTS / "Sleepiness_TimeSeries_Test.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] report -> {out_md}")
    return out_md


def compile_pdf(md):
    try:
        r = subprocess.run(["pandoc", md.name, "-o", "_ts_build.pdf", "--pdf-engine=xelatex",
                            "-V", "geometry:margin=1in"], cwd=str(DRAFTS), capture_output=True, text=True)
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
