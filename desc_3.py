"""
desc_3.py
================================================================================
Descriptive statistics: banking-conglomerate CLUSTER-SIZE IMBALANCE and DEPOSIT
CONCENTRATION. This is the single, canonical home for cluster-imbalance reporting
used to justify the wild cluster bootstrap (WCB) over cluster-robust / Delta-method
standard errors.

Everything is computed on the EXACT specification-12 second-stage estimation sample
we report (the pooled joint single-index estimator, E7), read from
    ESTIMATION_OUTPUT/DEMAND_PREP/est7/market_panel_phis.csv,
so the table's Observations / Clusters match the estimation tables (441,331 / 506).

Metrics (Panel A): nominal clusters G vs effective G* = G/(1+CV^2)
(Carter-Schnepel-Steigerwald 2017; via utils.cluster.effective_cluster_stats), the
coefficient of variation of cluster sizes (the quantity in MacKinnon-Webb 2017),
mean/median/max obs per cluster, and the deposit HHI with its numbers-equivalent
1/HHI. Panel B: the top-N conglomerates by time-averaged national deposit share, with
cumulative share and each cluster's observation share.

Outputs (CSV + LaTeX to ESTIMATION_OUTPUT/Rout, mirrored to Drafts/Deposit Competition):
    cluster_imbalance.tex   (\\input-able two-panel table for V_Main.tex)
    cluster_imbalance.csv
    cluster_imbalance.json  (the underlying numbers)

Usage:
    python desc_3.py                 # est7, top-5
    python desc_3.py --est 7 --top-n 10

NOTE: reads estimation output, so it runs AFTER the sleep estimation (unlike
desc_1/desc_2, which read the raw market panel).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as P
from utils.cluster import effective_cluster_stats

CLUSTER_VAR = "CodConglomeradoPrudencial"
_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = P.PROCESSED / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = _ROOT / "Drafts" / "Deposit Competition"

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
    path = P.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est}" / "market_panel_phis.csv"
    if not path.exists():
        raise FileNotFoundError(f"Second-stage sample not found: {path} (run the sleep estimation first).")
    df = pd.read_csv(path, usecols=[CLUSTER_VAR, "NomeInstituicao", "CNPJ_Lider",
                                    "deposit_balance", "time_id"])
    return df, path


def assert_matches_pkl(df, est):
    """Cross-check the sample against the estimator's saved fit so the exhibit is
    provably consistent with the reported Observations / Clusters."""
    pkl = P.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est}" / "estimation_results.pkl"
    try:
        import pickle
        ss = pickle.load(open(pkl, "rb")).get("IV_HausmanFull x Tech", {}).get("second_stage")
        nobs, G = int(getattr(ss, "nobs", -1)), int(getattr(ss, "G_nominal", -1))
        if nobs > 0 and nobs != len(df):
            print(f"  [WARN] rows {len(df):,} != pkl nobs {nobs:,}")
        if G > 0 and G != df[CLUSTER_VAR].nunique():
            print(f"  [WARN] clusters {df[CLUSTER_VAR].nunique()} != pkl G_nominal {G}")
        return nobs, G
    except Exception as e:
        print(f"  [note] could not cross-check vs pkl: {e}")
        return None, None


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


def render_tex(st):
    def pct(x):
        return f"{100 * x:.1f}\\%"
    N = f"{int(st['total_obs']):,}"
    L = [r"\begin{table}[ht]", r"\setstretch{1.0}", r"\centering",
         r"\caption{Conglomerate cluster-size imbalance and deposit concentration}",
         r"\label{tab:cluster_imbalance}", r"\footnotesize", r"\begin{threeparttable}",
         r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"\multicolumn{5}{l}{\textit{Panel A. Cluster structure (banking conglomerates)}} \\",
         r"\midrule",
         r"\multicolumn{4}{l}{Clusters (conglomerates), $G$} & " + f"{st['G_nominal']:,}" + r" \\",
         r"\multicolumn{4}{l}{Effective clusters, $G^{*}=G/(1+\mathrm{CV}^{2})$} & " + f"{st['G_star']:.1f}" + r" \\",
         r"\multicolumn{4}{l}{Total observations} & " + N + r" \\",
         r"\multicolumn{4}{l}{Obs.\ per cluster: mean / median / max} & "
         + f"{st['mean_size']:,.0f} / {st['median_size']:,.0f} / {st['max_size']:,.0f}" + r" \\",
         r"\multicolumn{4}{l}{Coefficient of variation of cluster sizes} & " + f"{st['cv']:.2f}" + r" \\",
         r"\multicolumn{4}{l}{Deposit HHI ($1/\mathrm{HHI}$)} & "
         + f"{st['hhi']:.3f} ({st['inv_hhi']:.1f})" + r" \\",
         r"\midrule",
         r"\multicolumn{5}{l}{\textit{Panel B. Five largest conglomerates by national deposit share}} \\",
         r"\midrule",
         r" & & Dep.\ share & Cumulative & Obs.\ share \\", r"\midrule"]
    for i, r in enumerate(st["top_n"], 1):
        L.append(f"{i} & {_latexify(r['name'])} & {pct(r['dep_share'])} & {pct(r['cum_share'])} & {pct(r['obs_share'])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]", r"\footnotesize",
          r"\item \textit{Notes:} Computed on the specification-12 second-stage estimation sample "
          r"(the pooled joint single-index estimator; " + N + r" observations across "
          + f"{st['G_nominal']}" + r" banking conglomerates). A cluster is a prudential conglomerate "
          r"(\texttt{CodConglomeradoPrudencial}); ``Obs.\ share'' is a conglomerate's share of "
          r"estimation observations (its cluster size) and ``Dep.\ share'' its time-averaged national "
          r"share of total deposits. The effective number of clusters $G^{*}=G/(1+\mathrm{CV}^{2})$ "
          r"\parencite{carter2017asymptotic} collapses to $\approx" + f"{st['G_star']:.0f}" + r"$ under "
          r"this size imbalance; with such unequal cluster sizes the cluster-robust (Delta-method) "
          r"$t$-test over-rejects, which motivates the wild cluster bootstrap "
          r"\parencite{cameron2008bootstrap,mackinnon2017wild}. Conglomerate names are resolved from "
          r"the leader institution's CNPJ.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", r"\doublespacing"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Cluster-imbalance / deposit-concentration descriptives.")
    ap.add_argument("--est", type=int, default=7, help="estimator whose second-stage sample to use (default 7 = joint sieve).")
    ap.add_argument("--top-n", type=int, default=5, help="conglomerates listed in Panel B (default 5).")
    args = ap.parse_args()

    print(f"Loading spec-12 second-stage sample (est{args.est})...")
    df, path = load_sample(args.est)
    print(f"  rows={len(df):,}  clusters={df[CLUSTER_VAR].nunique()}  ({path.name})")
    assert_matches_pkl(df, args.est)

    st, sizes = build_stats(df, top_n=args.top_n)
    print(f"  G={st['G_nominal']}  G*={st['G_star']:.2f}  CV={st['cv']:.2f}  HHI={st['hhi']:.3f} (1/HHI={st['inv_hhi']:.2f})")
    for i, r in enumerate(st["top_n"], 1):
        print(f"   {i}. {r['name']:<28s} dep={100*r['dep_share']:5.1f}%  cum={100*r['cum_share']:5.1f}%  obs={100*r['obs_share']:5.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # LaTeX table
    tex = render_tex(st)
    (OUTPUT_DIR / "cluster_imbalance.tex").write_text(tex, encoding="utf-8")
    (DRAFTS_DIR / "cluster_imbalance.tex").write_text(tex, encoding="utf-8")
    # CSV of the underlying numbers
    csv = pd.DataFrame(st["top_n"])
    csv.to_csv(OUTPUT_DIR / "cluster_imbalance.csv", index=False, encoding="utf-8")
    csv.to_csv(DRAFTS_DIR / "cluster_imbalance.csv", index=False, encoding="utf-8")
    # relocated cluster diagnostics (E7 sample)
    diag = {"est": args.est, "sample": "spec12_second_stage",
            "G_nominal": st["G_nominal"], "G_star": st["G_star"], "coefficient_variation": st["cv"],
            "mean_obs_per_cluster": st["mean_size"], "median_obs_per_cluster": st["median_size"],
            "max_obs_per_cluster": st["max_size"], "total_observations": int(st["total_obs"]),
            "deposit_hhi": st["hhi"], "inv_hhi": st["inv_hhi"],
            "top_n_by_deposit_share": st["top_n"]}
    (OUTPUT_DIR / "cluster_imbalance.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote cluster_imbalance.tex/.csv/.json to {OUTPUT_DIR} and mirrored to {DRAFTS_DIR}")


if __name__ == "__main__":
    main()
