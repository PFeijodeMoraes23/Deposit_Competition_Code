"""make_phi_t_band_table.py — national phi_t: level, movement, and honest uncertainty.

The estimation tables report a mean phi_t and nothing about how precisely it is pinned, so
the quantity the shape-constrained link exists to deliver -- a band that is valid when the
link sits on the boundary of its constraint set -- appears in no paper artifact. This builds
that table from the stored unconditional-band pickles.

WHICH BAND. `band_tangent_cone_total` is the reporting object: BOTH channels (the index
direction re-profiled at each draw, and the link fluctuation cone-projected at that draw's
own solution). The alternatives are not interchangeable and are shown as sub-rows because
the decomposition is itself the result:

  band_theta_only        direction channel alone, link frozen
  band_tangent_cone      link channel alone, evaluated at theta-hat
  band_total /           the PROJECTED construction. Reported here ONLY as the cautionary
  band_link_only         row: projecting a bootstrap draw onto {beta >= 0, sum beta <= 1}
                         is inconsistent when the estimator sits on the boundary (Andrews
                         2000), and it shows -- draws collapse onto the beta = 0 vertex,
                         which means phi identically 0, and the band blows out to ~99pp.

THE GATE. `method spread` is the agreement diagnostic: the gap between the percentile and
the bias-corrected interval width. Small means the map from coefficients to phi_t is
differentiable enough for the bootstrap to be reading a real sampling distribution; the
robust cells sit two orders of magnitude inside the 0.5pp benchmark, the LS cells fail it.

Reads and writes <rout_dir()>, so SLEEP_OUT_ROOT points it at a sandbox run.

  python make_phi_t_band_table.py                    # E3/E4, robust
  python make_phi_t_band_table.py --est 3 4 --loss robust ls
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import pickle
import shutil

import numpy as np
import pandas as pd

from utils import paths as _paths_mod
from utils import routines as _routines
from utils.sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)

_DRAFTS_DIR = _paths_mod.drafts_dir()
OUT_NAME = "tab_phi_t_bands_spec12.tex"

# Column header per estimator. These \ref the estimation-strategy enumerate in V_Main
# (sec:empirical:sleep), the same convention as every other sleepiness table, so a
# renumbering of the lineup cannot put a stale number in a caption. Both maps come from
# config/routines.toml: EST_LABEL is the enumerate slug the \ref wraps, EST_NAME the
# human-readable second header row.
EST_LABEL = {e: _routines.est_slug(e) for e in _routines.LINK_ESTS}
EST_NAME = {e: _routines.est_name(e) for e in _routines.LINK_ESTS}


def _width(df):
    """Mean 95% interval width in pp, or NaN if the band is absent."""
    if not isinstance(df, pd.DataFrame) or not {"lo", "hi"} <= set(df.columns):
        return float("nan")
    return float((df["hi"] - df["lo"]).mean() * 100.0)


def _method_spread(df):
    """Percentile-vs-bias-corrected width gap in pp -- the agreement gate."""
    if not isinstance(df, pd.DataFrame) or not {"lo_bc", "hi_bc"} <= set(df.columns):
        return float("nan")
    pct = (df["hi"] - df["lo"]) * 100.0
    bc = (df["hi_bc"] - df["lo_bc"]) * 100.0
    return float((pct - bc).abs().mean())


def load_cell(est, loss):
    p = _paths_mod.rout_dir() / f"ts_link_band_est{est}_uncond_{loss}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        obj = pickle.load(f)
    total = obj.get("band_tangent_cone_total")
    if not isinstance(total, pd.DataFrame):
        return None
    phi = total["phi_t"]
    meta = obj.get("meta", {}) or {}
    return dict(
        est=est, loss=loss,
        level=float(phi.mean() * 100.0),
        rng=float((phi.max() - phi.min()) * 100.0),
        band=_width(total),
        band_theta=_width(obj.get("band_theta_only")),
        band_link=_width(obj.get("band_tangent_cone")),
        band_proj=_width(obj.get("band_total")),
        spread=_method_spread(total),
        B=meta.get("B"), n_cl=meta.get("n_cl"), n=meta.get("n"),
        constrained=meta.get("constrained"),
    )


def _fmt(x, nd=2):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def build_tex(cells):
    k = len(cells)
    nobs = [f"{c['n']:,}" if c["n"] else "---" for c in cells]
    heads = " & ".join(rf"\ref{{estimation:{EST_LABEL.get(c['est'], '')}}}" for c in cells)
    names = " & ".join(EST_NAME.get(c["est"], f"E{c['est']}") for c in cells)
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{National Sleepiness: Level, Movement, and Uncertainty (Spec 12)}",
        r"\label{tab:phi_t_bands_spec12}",
        r"\setstretch{1.0}",
        rf"\begin{{tabular}}{{l{'c' * k}}}",
        r"\toprule",
        rf" & \multicolumn{{{k}}}{{c}}{{Estimation Strategy}} \\",
        rf"\cmidrule(lr){{2-{k + 1}}}",
        rf" & {heads} \\",
        rf" & {names} \\",
        r"\midrule",
        rf"Mean $\hat{{\phi}}_t$ (pp) & {' & '.join(_fmt(c['level']) for c in cells)} \\",
        rf"$\hat{{\phi}}_t$ range (pp) & {' & '.join(_fmt(c['rng']) for c in cells)} \\",
        r"\addlinespace",
        rf"95\% band width (pp)\tnote{{a}} & {' & '.join(_fmt(c['band']) for c in cells)} \\",
        rf"\quad direction channel & {' & '.join(_fmt(c['band_theta']) for c in cells)} \\",
        rf"\quad link channel & {' & '.join(_fmt(c['band_link']) for c in cells)} \\",
        r"\addlinespace",
        rf"Method spread (pp)\tnote{{b}} & {' & '.join(_fmt(c['spread'], 3) for c in cells)} \\",
        rf"Projected band (pp)\tnote{{c}} & {' & '.join(_fmt(c['band_proj'], 1) for c in cells)} \\",
        r"\midrule",
        rf"Bootstrap draws & {' & '.join(str(c['B'] or '---') for c in cells)} \\",
        rf"Clusters & {' & '.join(str(c['n_cl'] or '---') for c in cells)} \\",
        rf"Observations & {' & '.join(nobs)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item \textit{Notes:} Spec 12 (\texttt{IV\_HausmanFull} $\times$ Tech), "
        r"Cauchy-robust loss. $\hat{\phi}_t$ is the population-weighted mean of $\phi_{mt}$ "
        r"across markets, as in the definition of $\phi_t$ in Section~\ref{sec:model}. The "
        r"link is estimated under its shape constraints ($\beta \ge 0$, $\sum\beta \le 1$), "
        r"so the fitted $G$ is a genuine sub-CDF rather than a curve truncated after the fact.",
        r"\item[a] Both channels: the index direction is perturbed through its cluster "
        r"influence functions and the link is re-profiled at each draw, with the link "
        r"fluctuation projected onto the tangent cone of the active constraints at that "
        r"draw's own solution (\textcite{geyer1994}; \textcite{andrews2000inconsistency}; "
        r"\textcite{hongli2018}; \textcite{fangsantos2019}). WCB at the "
        r"conglomerate level (\textcite{cameron2008bootstrap}; "
        r"\textcite{mackinnon2017wild}), Webb six-point weights, WCU-$t$.",
        r"\item[b] Largest gap between the percentile and bias-corrected interval widths. "
        r"A small spread is evidence that the map from coefficients to $\hat{\phi}_t$ is "
        r"differentiable enough for the bootstrap to be consistent; the benchmark from the "
        r"linear estimators is $\le 0.5$pp.",
        r"\item[c] Reported only as a diagnostic. Projecting a bootstrap draw onto the "
        r"constraint set is inconsistent when the estimator lies on the boundary "
        r"(\textcite{andrews2000inconsistency}): draws collapse onto the $\beta = 0$ "
        r"vertex, at which $\phi \equiv 0$, and the interval degenerates. It is not a "
        r"confidence interval.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="National phi_t band table (spec 12).")
    _default = list(_routines.LINK_ESTS)
    ap.add_argument("--est", type=int, nargs="+", default=_default, choices=sorted(EST_LABEL),
                    help=f"estimators to report (default: {' '.join(map(str, _default))}, "
                         "the reported pair)")
    ap.add_argument("--loss", nargs="+", default=["robust"],
                    choices=("robust", "ls"), help="loss cells (default: robust only)")
    a = ap.parse_args()

    cells = []
    for loss in a.loss:
        for est in a.est:
            c = load_cell(est, loss)
            if c is None:
                print(f"  [skip] no band pickle for est{est}/{loss}")
                continue
            cells.append(c)
            print(f"  est{est}/{loss}: level={c['level']:.2f}pp range={c['rng']:.2f}pp "
                  f"band={c['band']:.2f}pp (theta {c['band_theta']:.2f} / link "
                  f"{c['band_link']:.2f}) spread={c['spread']:.3f}pp "
                  f"constrained={c['constrained']}")
    if not cells:
        print("[!] no band pickles found -- run sleep_band_uncond.py first.")
        return

    out = _DRAFTS_DIR / OUT_NAME
    out.write_text(build_tex(cells), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
