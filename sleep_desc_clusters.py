"""
sleep_desc_clusters.py
================================================================================
Descriptive statistics: banking-conglomerate CLUSTER-SIZE IMBALANCE and DEPOSIT
CONCENTRATION. This is the single, canonical home for cluster-imbalance reporting
used to justify the wild cluster bootstrap (WCB) over cluster-robust / Delta-method
standard errors.

Everything is computed on the EXACT specification-12 second-stage estimation sample we
report (the pooled single-index estimator, E3): the raw
    ESTIMATION_OUTPUT/DEMAND_PREP/est3/market_panel_phis.csv
minus the type-4/5 rows whose IV_HausmanFull instruments are missing (load_sample), so the
table's Observations / Clusters match the estimation tables (1,141,234 / 453). That match is
enforced, not just asserted in this docstring: assert_matches_pkl hard-fails if the sample
built from the CSV ever disagrees with estimation_results.pkl.

Metrics (Panel A): nominal clusters G vs effective G* = G/(1+CV^2)
(Carter-Schnepel-Steigerwald 2017; via utils.cluster_stats.effective_cluster_stats), the
coefficient of variation of cluster sizes (the quantity in MacKinnon-Webb 2017),
mean/median/max obs per cluster, and the deposit HHI with its numbers-equivalent
1/HHI. Panel B: the top-N conglomerates by time-averaged national deposit share, with
cumulative share and each cluster's observation share.

Outputs (CSV + LaTeX; Drafts/Deposit Competition locally, ESTIMATION_OUTPUT/Rout on the cluster):
    cluster_imbalance_panelA.tex  (\\input-able standalone table [H]: cluster structure / WCB justification; \\label{tab:cluster_imbalance})
    cluster_imbalance_panelB.tex  (\\input-able standalone table [H], spans \\textwidth: deposit concentration, top-N, for V_Main.tex; \\label{tab:deposit_concentration})
    cluster_imbalance.csv
    cluster_imbalance.json  (the underlying numbers)
Panels A and B are separate, self-contained tables (no cross-\\ref) so Panel B can be
\\input into V_Main.tex on its own.

Usage:
    python sleep_desc_clusters.py                 # est3, top-5
    python sleep_desc_clusters.py --est 3 --top-n 10

NOTE: reads estimation output, so it runs AFTER the sleep estimation (unlike
desc_1/desc_2, which read the raw market panel).
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd

from sleep_est_e2 import define_specifications
from utils import paths as P
from utils import routines as _routines
from utils.cluster_stats import effective_cluster_stats

CLUSTER_VAR = "CodConglomeradoPrudencial"
# rout_dir/est_dir follow SLEEP_OUT_ROOT, so the exhibit describes the sample of the run
# that produced it rather than whatever sits in the production tree.
OUTPUT_DIR = P.rout_dir()
DRAFTS_DIR = P.drafts_dir()

# ONE copy, chosen by environment. On the cluster only data/output/sleep/Rout is packaged by
# cluster_archive.sh, so a fragment written to the skeleton Drafts dir never reaches the
# download; locally the paper directory is the copy that matters and Rout would just be a
# second, divergeable one.
TEX_TARGETS = (OUTPUT_DIR,) if P.on_cluster() else (DRAFTS_DIR,)
# The .tex fragments are exhibits and follow TEX_TARGETS. The .csv/.json are the numbers
# BEHIND the table -- data, not exhibits -- so they stay in Rout on both sides, where the
# rest of the run's intermediates live and where the archiver already packages them.
DATA_DIR = OUTPUT_DIR

# 8-digit CNPJ root of the prudential-conglomerate leader -> display brand name.
# The panel's NomeInstituicao is frequently a subsidiary label with folded accents
# (e.g. "AGORA CORRETORA" for Bradesco), so recognizable names are resolved from the
# leader's CNPJ, which is authoritative. Extend as needed.
KNOWN_BANKS = {
    "00000000": "Banco do Brasil",
    "00360305": "Caixa Econômica Federal",
    "60872504": "Itaú Unibanco",
    "60701190": "Itaú Unibanco",
    "60746948": "Bradesco",
    "90400888": "Santander",
    "18236120": "Nubank",
    "30306294": "BTG Pactual",
    "58160789": "Safra",
    "92702067": "Banrisul",
    "59285411": "Banco Original",
    "17351180": "Banco Inter",
}

# Names are stored in Unicode (works natively in the xelatex report and in CSV/JSON);
# this escapes the common Portuguese accents to LaTeX for the pdflatex-compiled .tex,
# matching V_Main's convention (e.g. "Dep\'{o}sitos").
_ACCENTS = {"á": r"\'{a}", "â": r"\^{a}", "ã": r"\~{a}", "à": r"\`{a}",
            "é": r"\'{e}", "ê": r"\^{e}", "í": r"\'{i}", "ó": r"\'{o}",
            "ô": r"\^{o}", "õ": r"\~{o}", "ú": r"\'{u}", "ç": r"\c{c}",
            "Á": r"\'{A}", "Â": r"\^{A}", "É": r"\'{E}", "Ô": r"\^{O}"}


def _latexify(s):
    return "".join(_ACCENTS.get(ch, ch) for ch in str(s))


def _ordinal(i):
    """LaTeX ordinal with a raised suffix, e.g. 1 -> 1\\textsuperscript{st}."""
    suf = "th" if 10 <= (i % 100) <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}" + r"\textsuperscript{" + suf + "}"


def _display_name(cnpj_lider, fallback):
    """Recognizable conglomerate name from the leader CNPJ root; else the cleaned
    panel name."""
    root = None
    try:
        if pd.notna(cnpj_lider):
            root = str(int(float(cnpj_lider))).zfill(8)
    except (ValueError, TypeError):
        root = None
    if root in KNOWN_BANKS:
        return KNOWN_BANKS[root]
    return str(fallback).title() if pd.notna(fallback) else (root or "?")


def load_sample(est):
    """Spec-12 second-stage estimation sample: the raw CSV minus type-4/5 rows whose
    IV_HausmanFull instruments are missing. Those rows have an undefined control-function
    residual (v_hat) and are excluded from the second stage by
    utils.sleep_links.fit_nlls_link's `df.dropna(subset=cols + CF_cols)`, CF_cols =
    ['v_hat_x_lagged_dep']. v_hat itself isn't a column of this CSV, but a row's v_hat is
    undefined exactly when one of the instruments it is built from is missing, so the same
    exclusion is reproduced here from the instrument columns alone -- imported from the
    estimation code (define_specifications) rather than hardcoded.
    """
    path = P.est_dir(est) / "market_panel_phis.csv"
    if not path.exists():
        raise FileNotFoundError(f"Second-stage sample not found: {path} (run the sleep estimation first).")
    _, iv_specs, _ = define_specifications()
    hausman_cols = iv_specs["IV_HausmanFull"]
    df = pd.read_csv(path, usecols=[CLUSTER_VAR, "NomeInstituicao", "CNPJ_Lider",
                                    "deposit_balance", "time_id", "deposit_type"] + hausman_cols)
    drop = df["deposit_type"].isin([4, 5]) & df[hausman_cols].isna().any(axis=1)
    n_dropped = int(drop.sum())
    df = df.loc[~drop].reset_index(drop=True)
    return df, path, n_dropped


def assert_matches_pkl(df, est):
    """Cross-check the sample against the estimator's saved fit so the exhibit is provably
    consistent with the reported Observations / Clusters. Hard-fails (rather than warning):
    Panel B's cell values are wrong, not just cosmetically off, if the sample disagrees with
    the pickle the estimation tables are built from.
    """
    pkl = P.est_dir(est) / "estimation_results.pkl"
    ss = pickle.load(open(pkl, "rb")).get(_routines.SPEC12, {}).get("second_stage")
    nobs, G = int(getattr(ss, "nobs", -1)), int(getattr(ss, "G_nominal", -1))
    n_actual, g_actual = len(df), df[CLUSTER_VAR].nunique()
    assert nobs > 0 and nobs == n_actual, (
        f"rows {n_actual:,} != pkl nobs {nobs:,} (est{est}, spec {_routines.SPEC12!r})")
    assert G > 0 and G == g_actual, (
        f"clusters {g_actual} != pkl G_nominal {G} (est{est}, spec {_routines.SPEC12!r})")
    return nobs, G


def build_stats(df, top_n=5):
    sizes = df.groupby(CLUSTER_VAR).size()
    st = effective_cluster_stats(sizes.values)
    st["median_size"] = float(sizes.median())
    st["max_size"] = float(sizes.max())

    # time-averaged national deposit share per conglomerate: for each quarter, its
    # deposits / national deposits; averaged over quarters.
    g = df.dropna(subset=["deposit_balance"])
    dep_ct = g.groupby(["time_id", CLUSTER_VAR])["deposit_balance"].sum()
    tot_t = dep_ct.groupby(level="time_id").sum()
    share_ct = dep_ct / tot_t
    tavg = share_ct.groupby(level=CLUSTER_VAR).mean().sort_values(ascending=False)
    st["hhi"] = float((tavg ** 2).sum())
    st["inv_hhi"] = 1.0 / st["hhi"] if st["hhi"] > 0 else float("nan")

    nm = df.drop_duplicates(CLUSTER_VAR).set_index(CLUSTER_VAR)
    obs_sh = sizes / sizes.sum()
    rows, cum = [], 0.0
    for c in tavg.head(top_n).index:
        cum += float(tavg[c])
        rows.append({"code": c,
                     "name": _display_name(nm.loc[c, "CNPJ_Lider"], nm.loc[c, "NomeInstituicao"]),
                     "dep_share": float(tavg[c]), "cum_share": cum,
                     "obs_share": float(obs_sh[c])})
    st["top_n"] = rows
    return st, sizes


def _table_open(caption, label, full_width=False):
    """Standalone table preamble.

    Placement is [H] (exact position, `float` package) rather than [htbp]: these panels are
    referenced inline in the text and must not drift to the top of another page.

    full_width=True opens a threeparttable spanning \\textwidth, so the table fills the column
    instead of sitting as a narrow block. The tabular itself must then be a tabular* with
    \\extracolsep{\\fill} (see _tabular_full_width) — that spreads the COLUMNS to the full width
    rather than rescaling the type, which a \\resizebox would do.
    """
    width = r"\begin{threeparttable}[b]" if not full_width else r"\begin{threeparttable}"
    return [r"\begin{table}[H]", r"\setstretch{1.0}", r"\centering",
            r"\caption{" + caption + r"}", r"\label{" + label + r"}",
            r"\footnotesize", width]


def _tabular_full_width(colspec_body):
    """tabular* spanning \\textwidth. `colspec_body` is the column spec WITHOUT the leading
    @{} — e.g. 'l r r r'. \\extracolsep{\\fill} distributes the slack between columns."""
    return r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + colspec_body + r"@{}}"


def render_panel_a(st):
    """Standalone Panel A: conglomerate cluster-size imbalance (the WCB justification).
    Two-column (metric / value) table; self-contained (no cross-reference to Panel B)."""
    N = f"{int(st['total_obs']):,}"
    L = _table_open(r"Conglomerate cluster-size imbalance", r"tab:cluster_imbalance")
    L += [r"\begin{tabular}{@{}l r@{}}", r"\toprule",
          r"Clusters (conglomerates), $G$ & " + f"{st['G_nominal']:,}" + r" \\",
          r"Effective clusters, $G^{*}=G/(1+\mathrm{CV}^{2})$ & " + f"{st['G_star']:.1f}" + r" \\",
          r"Total observations & " + N + r" \\",
          r"Obs.\ per cluster: mean / median / max & "
          + f"{st['mean_size']:,.0f} / {st['median_size']:,.0f} / {st['max_size']:,.0f}" + r" \\",
          r"Coefficient of variation of cluster sizes & " + f"{st['cv']:.2f}" + r" \\",
          r"Deposit HHI ($1/\mathrm{HHI}$) & " + f"{st['hhi']:.3f} ({st['inv_hhi']:.1f})" + r" \\",
          r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]", r"\footnotesize",
          r"\item \textit{Notes:} Computed on the specification-12 second-stage estimation sample "
          r"(the pooled joint single-index estimator; " + N + r" observations across "
          + f"{st['G_nominal']}" + r" banking conglomerates). A cluster is a prudential conglomerate "
          r"(\texttt{CodConglomeradoPrudencial}). The effective number of clusters "
          r"$G^{*}=G/(1+\mathrm{CV}^{2})$ \parencite{carter2017asymptotic} collapses to $\approx"
          + f"{st['G_star']:.0f}" + r"$ under this size imbalance; with such unequal cluster sizes the "
          r"cluster-robust (Delta-method) $t$-test over-rejects, which motivates the wild cluster "
          r"bootstrap \parencite{cameron2008bootstrap,mackinnon2017wild}.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(L)


def render_panel_b(st):
    """Standalone Panel B: deposit concentration and cluster size among the largest
    conglomerates. Self-contained so it can be \\input on its own into V_Main.tex."""
    def pct(x):
        return f"{100 * x:.3f}\\%"
    n = len(st["top_n"])
    n_word = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
              9: "Nine", 10: "Ten"}.get(n, str(n))
    caption = f"Deposit Concentration and Cluster Size, {n_word} Largest Conglomerates"
    L = _table_open(caption, r"tab:deposit_concentration", full_width=True)
    L += [_tabular_full_width(r"l r r r"), r"\toprule",
          r" & Dep.\ share & Cumulative & Obs.\ share \\", r"\midrule"]
    for i, r in enumerate(st["top_n"], 1):
        L.append(f"{_ordinal(i)} & {pct(r['dep_share'])} & {pct(r['cum_share'])} & {pct(r['obs_share'])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular*}",
          r"\begin{tablenotes}[flushleft]", r"\footnotesize",
          r"\item \textit{Notes:} Specification-12 second-stage estimation sample ("
          + f"{st['G_nominal']:,}"
          + r" conglomerates), which excludes "
          + f"{st['n_dropped_hausman']:,}"
          + r" type-4/5 cells lacking the instruments of the control-function correction. "
          r"Conglomerates are ranked by their time-averaged national share of "
          r"total deposits (``Dep.\ share''). ``Obs.\ share'' is a conglomerate's share of estimation "
          r"observations (its cluster size).",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Cluster-imbalance / deposit-concentration descriptives.")
    ap.add_argument("--est", type=int, default=3, help="estimator whose second-stage sample to use (default 3 = single index, the reported routine).")
    ap.add_argument("--top-n", type=int, default=5, help="conglomerates listed in Panel B (default 5).")
    args = ap.parse_args()

    print(f"Loading spec-12 second-stage sample (est{args.est})...")
    df, path, n_dropped = load_sample(args.est)
    print(f"  rows={len(df):,}  clusters={df[CLUSTER_VAR].nunique()}  ({path.name}; "
          f"{n_dropped:,} type-4/5 rows dropped for missing IV_HausmanFull instruments)")
    assert_matches_pkl(df, args.est)

    st, sizes = build_stats(df, top_n=args.top_n)
    st["n_dropped_hausman"] = n_dropped
    print(f"  G={st['G_nominal']}  G*={st['G_star']:.2f}  CV={st['cv']:.2f}  HHI={st['hhi']:.3f} (1/HHI={st['inv_hhi']:.2f})")
    for i, r in enumerate(st["top_n"], 1):
        print(f"   {i}. {r['name']:<28s} dep={100*r['dep_share']:5.1f}%  cum={100*r['cum_share']:5.1f}%  obs={100*r['obs_share']:5.1f}%")

    # LaTeX tables: Panel A and Panel B as SEPARATE, standalone \input-able tables so
    # Panel B (deposit concentration) can be dropped into V_Main.tex on its own.
    tex_a, tex_b = render_panel_a(st), render_panel_b(st)
    for d in TEX_TARGETS:
        (d / "cluster_imbalance_panelA.tex").write_text(tex_a, encoding="utf-8")
        (d / "cluster_imbalance_panelB.tex").write_text(tex_b, encoding="utf-8")
        # drop the superseded combined two-panel table so no stale orphan remains
        (d / "cluster_imbalance.tex").unlink(missing_ok=True)
    # CSV of the underlying numbers
    csv = pd.DataFrame(st["top_n"])
    if True:
        csv.to_csv(DATA_DIR / "cluster_imbalance.csv", index=False, encoding="utf-8")
    # relocated cluster diagnostics (E3 sample)
    diag = {"est": args.est, "sample": "spec12_second_stage",
            "G_nominal": st["G_nominal"], "G_star": st["G_star"], "coefficient_variation": st["cv"],
            "mean_obs_per_cluster": st["mean_size"], "median_obs_per_cluster": st["median_size"],
            "max_obs_per_cluster": st["max_size"], "total_observations": int(st["total_obs"]),
            "deposit_hhi": st["hhi"], "inv_hhi": st["inv_hhi"],
            "top_n_by_deposit_share": st["top_n"]}
    (DATA_DIR / "cluster_imbalance.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] cluster_imbalance_panel{{A,B}}.tex -> {TEX_TARGETS[0]}; "
          f".csv/.json -> {DATA_DIR}")


if __name__ == "__main__":
    main()
