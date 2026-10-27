"""
make_active_truncation.py
=========================
Where the active-deposit construction (14-B) truncates at zero (routines E3, E4, specification
12), from the spec-12 cells of utils/sleep_cells.py (the stored sleep fit; needs the cluster's
sleepiness run). A cell is truncated when its balance is at most the carried-forward stock,
Dep <= phi-hat g Dep_lag, so its active balance is floored at zero and it leaves the demand sample.

  shares          of cells and of balances, overall and by firm type, deposit type, B market
                  population quintile and year, and among rows with phi-hat g >= 1
  bound violation the share of B market-quarters where phi-hat_mt exceeds min_j (or the 10th
                  percentile of) Dep / (g Dep_lag) across the market's B cells

phi-hat is phi-hat_mt for B firms and the national phi-hat_t for D firms. Diagnostic only: the
demand estimation keeps the construction as it is.

Outputs: Rout/active_truncation.json (keys E3, E4), read by make_paper_numbers.py
(demand.flows.neg.*); Drafts/Deposit Competition/tab_negative_active.tex.

Usage:  python make_active_truncation.py [--refresh]    (--refresh rebuilds the cached cells)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils.sleep_cells import ROUTINES, pre_cut
from make_xi_diagnostics import f3, compile_check

SIDECAR = _paths.rout_dir() / "active_truncation.json"
TEX = _paths.drafts_dir() / "tab_negative_active.tex"


def truncation(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["trunc"] = ~(d["active_raw"] > 1e-6)
    d["firm"] = np.where(d["is_B"].astype(bool), "B", "D")
    d["phi_row"] = np.where(d["is_B"].astype(bool), d["phi_mt"], d["phi_t"])

    def shares(f):
        return dict(share_cells=float(f["trunc"].mean()),
                    share_bal=float(f.loc[f["trunc"], "deposit_balance"].sum()
                                    / f["deposit_balance"].sum()),
                    n=int(len(f)))

    out = shares(d)
    out["by_firm"] = {k: shares(f) for k, f in d.groupby("firm")}
    out["by_type"] = {f"k{int(k)}": shares(f) for k, f in d.groupby("deposit_type")}
    out["by_year"] = {str(int(k)): shares(f) for k, f in d.groupby("year")}
    b = d[d["firm"] == "B"]
    pop = b.groupby("mca_code")["pop_total"].mean()
    q = pd.qcut(pop.rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)])
    bq = b.assign(size=b["mca_code"].map(q))
    out["by_size_B"] = {str(k): shares(f) for k, f in bq.groupby("size", observed=True)}
    pg = d["phi_row"] * d["gross_return_lag"] >= 1.0
    out["phig_ge1"] = dict(share_rows=float(pg.mean()), **shares(d[pg]))
    # Market bound: phi_mt <= min_j Dep / (g Dep_lag) over the B rows of a market-quarter.
    den = b["gross_return_lag"] * b["lagged_deposits"]
    r = (b["deposit_balance"] / den).where(den > 0)
    mb = b.assign(r=r).dropna(subset=["r"]).groupby(["mca_code", "time_id"]).agg(
        phi=("phi_mt", "first"), rmin=("r", "min"), rp10=("r", lambda x: x.quantile(0.10)),
        nrows=("r", "size"))
    out["bound_violation"] = dict(min=float((mb["phi"] > mb["rmin"]).mean()),
                                  p10=float((mb["phi"] > mb["rp10"]).mean()),
                                  n_market_quarters=int(len(mb)))
    return out


def _rt(e: int) -> str:
    from utils import routines as R
    r = R.ROUTINES[e]
    return f"{r.tex_name} {r.tex_ref}"


def table(res: dict) -> str:
    ests = [e for e in ROUTINES if f"E{e}" in res]

    def row(label, get):
        cells = []
        for e in ests:
            s = get(res[f"E{e}"])
            cells += [f3(s["share_cells"]), f3(s["share_bal"])]
        return f"\\quad {label} & " + " & ".join(cells) + r" \\"

    def panel(t):
        return rf"\multicolumn{{{1 + 2 * len(ests)}}}{{@{{}}l}}{{\textit{{{t}}}}} \\"

    r0 = res[f"E{ests[0]}"]
    body = [row("All cells", lambda s: s), r"\addlinespace", panel("By firm type")]
    body += [row(f"{k} firms", lambda s, k=k: s["by_firm"][k]) for k in ("B", "D")]
    body += [r"\addlinespace", panel("By deposit type")]
    names = {"k1": "Demand", "k2": "Savings", "k4": "Time", "k5": "Prepaid"}
    body += [row(names[k], lambda s, k=k: s["by_type"][k]) for k in names if k in r0["by_type"]]
    body += [r"\addlinespace", panel("B firms, by market population quintile")]
    body += [row(f"Quintile {k[1]}", lambda s, k=k: s["by_size_B"][k]) for k in sorted(r0["by_size_B"])]
    body += [r"\addlinespace", panel("By year")]
    body += [row(y, lambda s, y=y: s["by_year"][y]) for y in sorted(r0["by_year"])]
    body += [r"\addlinespace", row(r"Rows with $\hat{\phi}g \geq 1$", lambda s: s["phig_ge1"])]
    viol = []
    for e in ests:
        v = res[f"E{e}"]["bound_violation"]
        viol += [f3(v["min"]), f3(v["p10"])]
    shr = []
    for e in ests:
        shr += [f3(res[f"E{e}"]["phig_ge1"]["share_rows"]), ""]
    body += [r"\quad Share of rows with $\hat{\phi}g \geq 1$ & " + " & ".join(shr) + r" \\",
             r"\midrule",
             r"Market-quarters with $\hat{\phi}_{mt}$ above the bound & \multicolumn{"
             + str(2 * len(ests)) + r"}{c}{(minimum, 10th percentile)} \\",
             r"\quad Share & " + " & ".join(viol) + r" \\"]
    head1 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{_rt(e)}}}" for e in ests)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ests)))
    head2 = " & " + " & ".join("Cells & Balances" for _ in ests)
    note = (r"\textit{Notes:} Cells (conglomerate $\times$ deposit type $\times$ minimum comparable "
            r"area (MCA) $\times$ quarter, specification 12) whose active balance in the "
            r"active-deposit construction, the balance minus the carried-forward stock "
            r"$\hat{\phi}\,g\,\text{Dep}_{t-1}$, is at most zero. The construction truncates them at "
            r"zero and they leave the demand sample. Shares of all cells and of all balances in each group. "
            r"$\hat{\phi}$ is $\hat{\phi}_{mt}$ for brick-and-mortar (B) firms and the national "
            r"$\hat{\phi}_t$ for digital (D) firms. Market population quintiles: B firms' MCAs by "
            r"mean population. Last panel: share of B market-quarters where $\hat{\phi}_{mt}$ "
            r"exceeds the bound the model implies, $\min_j \text{Dep}_{jkmt}/(g_{jkm,t-1}\,"
            r"\text{Dep}_{jkm,t-1})$ across the market's B cells, and the same share with the "
            r"10th percentile in place of the minimum. Diagnostic only: the demand estimation "
            r"keeps the active-deposit construction as it is.")
    return "\n".join([
        "% tab_negative_active.tex -- GENERATED by make_active_truncation.py. Do not edit.",
        r"\begin{spacing}{1.0}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}\captionsetup{position=top}",
        r"\captionof{table}{Cells with Negative Active Demand}"
        r"\label{tab:negative_active}",
        r"\begin{tabularx}{\textwidth}{@{}X"
        + r">{\raggedleft\arraybackslash}p{1.7cm}" * (2 * len(ests)) + r"@{}}",
        r"\toprule", head1 + r" \\", rules, head2 + r" \\", r"\midrule", *body, r"\bottomrule",
        rf"\multicolumn{{{1 + 2 * len(ests)}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {note}}} \\",
        r"\end{tabularx}", r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
        r"\end{spacing}", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached spec-12 cells")
    a = ap.parse_args()
    res, meta = {}, {}
    for e in ROUTINES:
        d, _ups, key = pre_cut(e, a.refresh)
        meta[f"E{e}_inputs"] = key
        c = res[f"E{e}"] = truncation(d)
        print(f"E{e} cells {c['share_cells']:.3f} bal {c['share_bal']:.3f} "
              f"bound {c['bound_violation']} phig>=1 {c['phig_ge1']['share_rows']:.3f}"
              f"/{c['phig_ge1']['share_cells']:.3f}")
    meta["written"] = datetime.datetime.now().isoformat(timespec="seconds")
    SIDECAR.write_text(json.dumps(dict(_meta=meta, **res), indent=2, default=float),
                       encoding="utf-8")
    print(f"  wrote {SIDECAR}")
    TEX.write_text(table(res), encoding="utf-8")
    print(f"  wrote {TEX}")
    import make_xi_diagnostics as X
    X._STANDALONE = X._STANDALONE.replace("tab_xi_diagnostics.tex", TEX.name)
    if not compile_check(TEX.parent):
        sys.exit(f"{TEX.name} did not compile cleanly on its own")


if __name__ == "__main__":
    main()
